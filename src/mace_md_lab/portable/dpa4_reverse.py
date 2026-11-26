"""Fixed-weight FP32 DPA4-Mini message kernels for first-order MD inference.

All tensor layouts are qualified at installation for the (lmax=2, mmax=1)
single-focus checkpoint. Geometry and graph indices are inputs to every call.
"""
from contextlib import contextmanager

import torch
from torch.autograd.function import once_differentiable
import triton
import triton.language as tl


@triton.jit
def _attention_vjp(V, OUT, G, ALPHA, ENV, DST, DV, DL, DE, WIDTH: tl.constexpr,
                   BLOCK: tl.constexpr):
    e = tl.program_id(0)
    c = tl.arange(0, BLOCK)
    mask = c < WIDTH
    i = tl.load(DST + e).to(tl.int32)
    a = tl.load(ALPHA + e)
    env = tl.load(ENV + e)
    grad = tl.load(G + i * WIDTH + c, mask, 0.)
    value = tl.load(V + e * WIDTH + c, mask, 0.)
    mean = tl.load(OUT + i * WIDTH + c, mask, 0.)
    dot = a * tl.sum(grad * (value - mean), axis=0)
    tl.store(DV + e * WIDTH + c, a * grad, mask)
    tl.store(DL + e, dot)
    tl.store(DE + e, tl.where(env > 0., 2. * dot / tl.where(env > 0., env, 1.), 0.))


class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, logits, envelope, dst, null_bias, eps, nodes):
        from deepmd.pt.model.descriptor.sezm_nn.attention import segment_envelope_gated_softmax
        values, logits, envelope, dst = (v.contiguous() for v in (values, logits, envelope, dst))
        alpha = segment_envelope_gated_softmax(logits, envelope, dst, nodes, null_bias, eps)
        out = values.new_zeros(nodes, *values.shape[1:])
        out.index_add_(0, dst, values * alpha.reshape(-1, 1, 1))
        ctx.save_for_backward(values, out, alpha, envelope, dst)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad):
        v, out, a, env, dst = ctx.saved_tensors
        dv, dl, de = torch.empty_like(v), torch.empty_like(a), torch.empty_like(env)
        width = v.shape[1] * v.shape[2]
        if dst.numel():
            _attention_vjp[(len(dst),)](v,out,grad.contiguous(),a,env,dst,dv,dl,de,width,
                                       triton.next_power_of_2(width))
        return dv, dl, de, None, None, None, None


@triton.jit
def _mix_sub(X,K,G,Y,e,c,b,width:tl.constexpr,off:tl.constexpr,ko:tl.constexpr,REVERSE:tl.constexpr):
    for a in tl.static_range(width):
        accum = tl.full((32,),0.,tl.float32)
        for z in tl.static_range(width):
            if REVERSE:
                coeff = tl.load(K+e*13+ko+a*width+z)
                inp = tl.load(G+(e*7+off+z)*32+c)
            else:
                coeff = tl.load(K+e*13+ko+z*width+a)
                inp = tl.load(X+(e*7+off+z)*32+c)
            accum += inp*coeff
        tl.store(Y+(e*7+off+a)*32+c,accum*b)


@triton.jit
def _rank_one(X, K, BASIS, G, Y, DK, REVERSE: tl.constexpr):
    e = tl.program_id(0)
    c = tl.arange(0,32)
    b = tl.load(BASIS+c)
    # Three m=0 degrees; the other two signed-m blocks share a two-degree matrix.
    _mix_sub(X,K,G,Y,e,c,b,3,0,0,REVERSE)
    _mix_sub(X,K,G,Y,e,c,b,2,3,9,REVERSE)
    _mix_sub(X,K,G,Y,e,c,b,2,5,9,REVERSE)
    if REVERSE:
        for i in tl.static_range(3):
            for o in tl.static_range(3):
                x = tl.load(X+(e*7+i)*32+c)
                g = tl.load(G+(e*7+o)*32+c)
                tl.store(DK+e*13+i*3+o,tl.sum(x*(b*g),0))
        for i in tl.static_range(2):
            for o in tl.static_range(2):
                xn=tl.load(X+(e*7+3+i)*32+c)
                xp=tl.load(X+(e*7+5+i)*32+c)
                gn=tl.load(G+(e*7+3+o)*32+c)
                gp=tl.load(G+(e*7+5+o)*32+c)
                tl.store(DK+e*13+9+i*2+o,tl.sum((xn*gn+xp*gp)*b,0))


class _RankOne(torch.autograd.Function):
    """Mix degrees using the 13 learned coefficients and shared channel basis."""
    @staticmethod
    def forward(ctx,x,k,b):
        x,k,b=x.contiguous(),k.contiguous(),b.detach().contiguous()
        out=torch.empty_like(x)
        if len(x):
            _rank_one[(len(x),)](x,k,b,x,out,k,False,num_warps=1)
        ctx.save_for_backward(x,k,b)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        x,k,b=ctx.saved_tensors
        dx,dk=torch.empty_like(x),torch.empty_like(k)
        if len(x):
            _rank_one[(len(x),)](x,k,b,grad.contiguous(),dx,dk,True,num_warps=1)
        return dx,dk,None


@triton.jit
def _gate_kernel(X, LOG, G, Y, DL, E, REV:tl.constexpr, NE:tl.constexpr):
    e=tl.program_id(0)*NE+tl.arange(0,NE)
    c=tl.arange(0,32)
    ok=e[:,None]<E
    xc=X+e[:,None]*224+c[None,:]
    yc=Y+e[:,None]*224+c[None,:]
    z=tl.load(xc,ok,0.)
    s=tl.sigmoid(z)
    if REV:
        gc=G+e[:,None]*224+c[None,:]
        g0=tl.load(gc,ok,0.)
        tl.store(yc,g0*s*(1.+z*(1.-s)),ok)
    else:
        tl.store(yc,z*s,ok)
    for j in tl.static_range(2):
        idx=e[:,None]*64+j*32+c[None,:]
        a=tl.sigmoid(tl.load(LOG+idx,ok,0.))
        dlog=tl.full((NE,32),0.,tl.float32)
        for k in tl.static_range(3):
            d=1+j+2*k
            if REV:
                dy=tl.load(gc+d*32,ok,0.)
                dlog+=tl.load(xc+d*32,ok,0.)*dy
                tl.store(yc+d*32,dy*a,ok)
            else:
                tl.store(yc+d*32,tl.load(xc+d*32,ok,0.)*a,ok)
        if REV:
            tl.store(DL+idx,dlog*a*(1.-a),ok)


class _Gate(torch.autograd.Function):
    """Differentiate each degree gate once across all of its angular components."""
    @staticmethod
    def forward(ctx,x,w,wt,bias):
        x=x.contiguous()
        n=x.shape[1]
        z=x[0,:,0,:] @ w
        if bias is not None:
            z+=bias
        y=torch.empty_like(x)
        if n:
            _gate_kernel[(triton.cdiv(n,4),)](x,z,x,y,z,n,False,4)
        ctx.save_for_backward(x,z,wt)
        return y

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,z,wt=ctx.saved_tensors
        n=x.shape[1]
        dx,dl=torch.empty_like(x),torch.empty_like(z)
        if n:
            _gate_kernel[(triton.cdiv(n,4),)](x,z,g.contiguous(),dx,dl,n,True,4)
            dx[0,:,0,:].add_(dl @ wt)
        return dx,None,None,None


class _Linear(torch.autograd.Function):
    """Use fixed block matrices and their transposes in first-order inference."""
    @staticmethod
    def forward(ctx,x,weights,bias):
        ctx.weights=weights
        ctx.shape=x.shape
        n=x.shape[1]
        flat=x.reshape(n,-1)
        parts=[flat[:,i:j] @ w for i,j,_,_,w,_ in weights]
        out=torch.cat(parts,-1).reshape(1,n,x.shape[2],-1)
        if bias is not None:
            out[:,:,0,:].add_(bias)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        flat=grad.reshape(grad.shape[1],-1)
        parts=[flat[:,o:p] @ wt for _,_,o,p,_,wt in ctx.weights]
        return torch.cat(parts,-1).reshape(ctx.shape),None,None


@triton.jit
def _logits(Q,K,R,B,ENV,SRC,DST,LOG,N,NE:tl.constexpr):
    e=tl.program_id(0)*NE+tl.arange(0,NE)
    c=tl.arange(0,32)
    ok=e<N
    i=tl.load(DST+e,ok,0).to(tl.int32)
    j=tl.load(SRC+e,ok,0).to(tl.int32)
    q=tl.load(Q+i[:,None]*32+c[None,:],ok[:,None],0.)
    k=tl.load(K+j[:,None]*32+c[None,:],ok[:,None],0.)
    r=tl.load(R+e[:,None]*32+c[None,:],ok[:,None],0.)
    bias=tl.load(B+c)
    env=tl.load(ENV+e,ok,0.)
    dot=tl.sum(q*k,1)*0.1767766952966369+tl.sum(r*bias[None,:],1)
    log=tl.where(env>0.,dot+2.*tl.log(tl.where(env>0.,env,1.)),-float("inf"))
    tl.store(LOG+e,log,ok)


@triton.jit
def _normalize(LOG,PTR,ORDER,NULL,ALPHA,BS:tl.constexpr):
    n=tl.program_id(0)
    a=tl.load(PTR+n).to(tl.int32)
    end=tl.load(PTR+n+1).to(tl.int32)
    c=tl.arange(0,BS)
    null=tl.load(NULL)
    maximum=null
    for t in range(a,end,BS):
        ix=t+c
        e=tl.load(ORDER+ix,ix<end,0).to(tl.int32)
        value=tl.load(LOG+e,ix<end,-float("inf"))
        maximum=tl.maximum(maximum,tl.max(value,0))
    denom=tl.exp(null-maximum)
    for t in range(a,end,BS):
        ix=t+c
        e=tl.load(ORDER+ix,ix<end,0).to(tl.int32)
        value=tl.load(LOG+e,ix<end,-float("inf"))
        denom+=tl.sum(tl.exp(value-maximum),0)
    for t in range(a,end,BS):
        ix=t+c
        e=tl.load(ORDER+ix,ix<end,0).to(tl.int32)
        value=tl.load(LOG+e,ix<end,-float("inf"))
        tl.store(ALPHA+e,tl.exp(value-maximum)/denom,ix<end)


@triton.jit
def _segment(V,ALPHA,PTR,ORDER,OUT,BE:tl.constexpr,BC:tl.constexpr):
    n=tl.program_id(0)
    c=tl.program_id(1)*BC+tl.arange(0,BC)
    a=tl.load(PTR+n).to(tl.int32)
    stop=tl.load(PTR+n+1).to(tl.int32)
    val=tl.full((BC,),0.,tl.float32)
    for t in range(a,stop,BE):
        ix=t+tl.arange(0,BE)
        e=tl.load(ORDER+ix,ix<stop,0).to(tl.int32)
        alpha=tl.load(ALPHA+e,ix<stop,0.)
        v=tl.load(V+e[:,None]*288+c[None,:],(ix[:,None]<stop)&(c[None,:]<288),0.)
        val+=tl.sum(alpha[:,None]*v,0)
    tl.store(OUT+n*288+c,val,c<288)


@triton.jit
def _transport_vjp(L,ROT,SCALE,OUT,G,ALPHA,ENV,Q,K,B,SRC,DST,DL,DROT,DQ,DK,DR,DE):
    # G is a receiver adjoint. Contract it in the local frame once per edge.
    e=tl.program_id(0)
    j=tl.load(SRC+e).to(tl.int32)
    i=tl.load(DST+e).to(tl.int32)
    c=tl.arange(0,32)
    o=tl.arange(0,16)
    a=tl.load(ALPHA+e)
    go=tl.load(G+(i*9+o[:,None])*32+c[None,:],o[:,None]<9,0.)
    out=tl.load(OUT+(i*9+o[:,None])*32+c[None,:],o[:,None]<9,0.)
    s=tl.load(SCALE+o,o<9,0.)
    gs=go*s[:,None]
    # The node mean in the softmax derivative includes its physical null mass.
    delta=-tl.sum(tl.sum(go*out,1),0)
    for h in tl.static_range(7):
        loc=tl.load(L+(e*7+h)*32+c)
        rot=tl.load(ROT+(e*9+o)*7+h,o<9,0.)
        adj=tl.sum(gs*rot[:,None],0)
        tl.store(DL+(e*7+h)*32+c,a*adj)
        tl.store(DROT+(e*9+o)*7+h,a*tl.sum(gs*loc[None,:],1),o<9)
        delta+=tl.sum(adj*loc,0)
    dz=a*delta
    env=tl.load(ENV+e)
    tl.store(DE+e,tl.where(env>0.,2.*dz/tl.where(env>0.,env,1.),0.))
    q=tl.load(Q+i*32+c)
    k=tl.load(K+j*32+c)
    tl.atomic_add(DQ+i*32+c,k*(dz*0.1767766952966369),sem='relaxed')
    tl.atomic_add(DK+j*32+c,q*(dz*0.1767766952966369),sem='relaxed')
    tl.store(DR+e*32+c,tl.load(B+c)*dz)


class _Transport(torch.autograd.Function):
    """Receiver-segmented softmax and aggregation, with an edge-owned reverse."""
    @staticmethod
    def forward(ctx,local,rot,scale,q,k,r,b,env,src,dst,log_null,indices):
        l,rot,q,k,r,env,src,dst=(t.contiguous() for t in (local,rot,q,k,r,env,src,dst))
        n=len(q)
        sorted_dst,order=dst.sort()
        ptr=torch.searchsorted(sorted_dst,indices)
        logits=torch.empty_like(env)
        alpha=torch.empty_like(env)
        if len(dst):
            _logits[(triton.cdiv(len(dst),8),)](q,k,r,b,env,src,dst,logits,len(dst),8)
        _normalize[(n,)](logits,ptr,order,log_null,alpha,64)
        value=torch.bmm(rot,l)*scale.view(1,9,1)
        out=q.new_empty((n,9,32))
        _segment[(n,9)](value,alpha,ptr,order,out,16,32)
        ctx.save_for_backward(l,rot,scale,out,alpha,env,q,k,b,src,dst)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        l,rot,scale,out,alpha,env,q,k,b,src,dst=ctx.saved_tensors
        dl,drot=torch.empty_like(l),torch.empty_like(rot)
        dq,dk=torch.zeros_like(q),torch.zeros_like(k)
        dr=q.new_empty((len(dst),32))
        de=torch.empty_like(env)
        if len(dst):
            _transport_vjp[(len(dst),)](l,rot,scale,out,g.contiguous(),alpha,env,q,k,b,src,dst,
                                        dl,drot,dq,dk,dr,de,num_warps=4)
        return dl,drot,None,dq,dk,dr,None,de,None,None,None,None


@triton.jit
def _local_vjp(X,R,G,SRC,DX,DR):
    e=tl.program_id(0)
    o=tl.arange(0,16)
    c=tl.arange(0,32)
    j=tl.load(SRC+e).to(tl.int32)
    x=tl.load(X+(j*9+o[:,None])*32+c[None,:],o[:,None]<9,0.)
    dx=tl.full((16,32),0.,tl.float32)
    for h in tl.static_range(7):
        g=tl.load(G+(e*7+h)*32+c)
        r=tl.load(R+(e*7+h)*9+o,o<9,0.)
        dx+=r[:,None]*g[None,:]
        tl.store(DR+(e*7+h)*9+o,tl.sum(x*g[None,:],1),o<9)
    tl.atomic_add(DX+(j*9+o[:,None])*32+c[None,:],dx,o[:,None]<9,sem="relaxed")


class _LocalRotation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,rotation,src):
        x,rotation,src=(v.contiguous() for v in (x,rotation,src))
        ctx.save_for_backward(x,rotation,src)
        return torch.bmm(rotation,x.index_select(0,src))

    @staticmethod
    @once_differentiable
    def backward(ctx,g):
        x,r,src=ctx.saved_tensors
        dx,dr=torch.zeros_like(x),torch.empty_like(r)
        if len(src):
            _local_vjp[(len(src),)](x,r,g.contiguous(),src,dx,dr)
        return dx,dr,None


def _local_message(m,calls):
    from deepmd.pt.model.descriptor.sezm_nn.indexing import project_D_to_m
    if (m.node_wise_grid_product is not None or m.radial_hidden_proj is not None or m.use_so2_attn_res
        or m.focus_compete and m.n_focus>1 or any(l.bias0 is not None for l in m.so2_linears)):
        raise ValueError("local path needs the ungated rank-one Mini mixing stack")
    mixer=m.radial_degree_mixer
    def local(x,cache,radial_feat,return_local=False):
        if not return_local:
            raise ValueError("local path requires fused rotate-back")
        d=project_D_to_m(cache.D_full,m.coeff_index_m,m.ebed_dim_full,cache.D_to_m_cache,m.lmax,m.mmax)
        u=_LocalRotation.apply(x,d,cache.src)
        k=radial_feat.reshape(len(u),96) @ mixer.weight
        y=_RankOne.apply(u,k,mixer.channel_basis).unsqueeze(0)
        for i,(lin,norm,act) in enumerate(zip(m.so2_linears,m.so2_inter_norms,m.non_linearities)):
            delta=act(lin(norm(y)))
            if m.layer_scale:
                delta=delta*m.adam_so2_layer_scales[i].reshape(1,1,1,32)
            y=y+delta
        calls["local"]+=1
        return y.transpose(0,1),radial_feat
    return local


class _StockForwardTransport(torch.autograd.Function):
    """Use native score, rotation, softmax and reduction operations with our VJP."""
    @staticmethod
    def forward(ctx,local,rot,scale,q,k,r,b,env,src,dst,null_bias,eps):
        from deepmd.pt.model.descriptor.sezm_nn.attention import segment_envelope_gated_softmax
        l,rot,q,k,r,env,src,dst=(v.contiguous() for v in (local,rot,q,k,r,env,src,dst))
        value=torch.bmm(rot,l)*scale.view(1,9,1)
        radial_bias=torch.einsum("efi,ifo->efo",r[:,None,:],b.reshape(32,1,1))
        scores=(q.index_select(0,dst)*k.index_select(0,src)).sum(-1).reshape(-1,1,1)*(32**-.5)+radial_bias
        a=segment_envelope_gated_softmax(scores,env,dst,len(q),null_bias,eps)
        out=q.new_zeros((len(q),9,32))
        out.index_add_(0,dst,value*a.reshape(-1,1,1))
        ctx.save_for_backward(l,rot,scale,out,a,env,q,k,b,src,dst)
        return out

    backward = staticmethod(_Transport.backward)


def _transport_forward(m,calls,native_forward=False):
    from deepmd.pt.model.descriptor.sezm_nn.indexing import project_Dt_from_m
    bias=m.adamw_attn_logit_w.detach().reshape(32).contiguous()
    log_null=(torch.nn.functional.softplus(m.adamw_attn_z_bias_raw.detach())+m.eps).log().contiguous()
    scale=m.rotate_inv_rescale_full.detach().reshape(9).contiguous()
    ranges={}
    def forward(x,cache,radial_feat):
        if cache.edge_src_gate is not None:
            raise ValueError("requires a field-free ungated source")
        x=m.pre_focus_mix(x.unsqueeze(2)).squeeze(2)
        local,rad=m.so2_message(x,cache,radial_feat,return_local=True)
        rot=project_Dt_from_m(cache.Dt_full,m.coeff_index_m,m.ebed_dim_full,
                              cache.Dt_from_m_cache,m.lmax,m.mmax)
        scalar=x[:,0,:].reshape(len(x),1,32)
        qk=m.attn_qk_norm(scalar)
        q=m.attn_q_proj(qk).reshape(-1,32)
        k=m.attn_k_proj(qk).reshape(-1,32)
        if native_forward:
            out=_StockForwardTransport.apply(local.reshape(-1,7,32),rot,scale,q,k,rad[:,0,:].reshape(-1,32),
                                             bias,cache.edge_env,cache.src,cache.dst,m.adamw_attn_z_bias_raw,m.eps)
        else:
            key=(len(x),cache.dst.device,cache.dst.dtype)
            if key not in ranges:
                ranges[key]=torch.arange(len(x)+1,device=cache.dst.device,dtype=cache.dst.dtype)
            out=_Transport.apply(local.reshape(-1,7,32),rot,scale,q,k,rad[:,0,:].reshape(-1,32),bias,
                                 cache.edge_env,cache.src,cache.dst,log_null,ranges[key])
        gate=torch.sigmoid(torch.einsum("nfi,ifo->nfo",m.attn_output_gate_norm(scalar),m.adamw_attn_gate_w))
        out=out*gate.reshape(len(x),1,1)
        if m.message_node_grid_product is not None:
            out=out+m.message_node_grid_product(out,x)
        calls["transport"]+=1
        return m.post_focus_mix(out.unsqueeze(2)).squeeze(2)
    return forward


def _edge_forward(m,calls):
    def forward(x,edge_cache,radial_feat):
        if edge_cache.edge_src_gate is not None:
            raise ValueError("source propagation gate is outside the qualification")
        n=len(x)
        x=m.pre_focus_mix(x.unsqueeze(2)).squeeze(2)
        message,rad=m.so2_message(x,edge_cache,radial_feat)
        scalar=x[:,0,:].reshape(n,1,32)
        qk=m.attn_qk_norm(scalar)
        q=m.attn_q_proj(qk).index_select(0,edge_cache.dst)
        k=m.attn_k_proj(qk).index_select(0,edge_cache.src)
        bias=torch.einsum("efi,ifo->efo",rad[:,0,:].reshape(-1,1,32),m.adamw_attn_logit_w)
        logits=(q*k).sum(-1,keepdim=True)*(32**-.5)+bias
        base=_Attention.apply(message,logits,edge_cache.edge_env,edge_cache.dst,
                              m.adamw_attn_z_bias_raw,m.eps,n)
        gate=torch.sigmoid(torch.einsum("nfi,ifo->nfo",m.attn_output_gate_norm(scalar),m.adamw_attn_gate_w))
        out=base*gate.reshape(n,1,1)
        if m.message_node_grid_product is not None:
            out=out+m.message_node_grid_product(out,x)
        calls["edge"]+=1
        return m.post_focus_mix(out.unsqueeze(2)).squeeze(2)
    return forward


@contextmanager
def use_adjoint(blocks,variant="local"):
    """Install one ablation on these evaluated modules and restore them on exit."""
    message_edits=[]
    edits=[]
    caches=[]
    calls={"edge":0,"mixer":0,"linear":0,"gate":0,"transport":0,"local":0}
    def bind(m,fn):
        edits.append((m,m.__dict__.get("forward")))
        m.forward=fn
    def mix(m):
        def run(x,rad):
            calls["mixer"]+=1
            return _RankOne.apply(x,m._project_radial(rad),m.channel_basis)
        return run
    def gate(m):
        if (m.layout != "fndc" or m.scalar_act.activation != "silu" or
                m.expand_index.tolist() != [0,1,0,1,0,1]):
            raise ValueError("requires the seven-coefficient SiLU degree gate")
        w=m.gate_linear.weight.detach().reshape(32,64).contiguous()
        wt=w.T.contiguous()
        b=m.gate_linear.bias
        def run(x):
            calls["gate"]+=1
            return _Gate.apply(x,w,wt,None if b is None else b.detach().view(1,-1))
        return run
    def lin(m):
        with torch.no_grad():
            w=m._build_so2_weight()[:,0,:]
            weights=[(i,j,o,p,w[i:j,o:p].contiguous(),w[i:j,o:p].T.contiguous())
                     for i,j,o,p in m._block_diag_slices]
        def run(x):
            out=_Linear.apply(x,weights,m.bias0 if m.mlp_bias else None)
            calls["linear"]+=1
            return out
        return run
    try:
        if variant not in ("edge","linear","combined","full","packed","transport","local","reverse"):
            raise ValueError(variant)
        for _,m in blocks:
            if (m.training or m.dtype!=torch.float32 or m.lmax!=2 or m.mmax!=1 or m.n_atten_head!=1
                or m.n_focus!=1 or m.hidden_channels!=32 or not m.needs_local_frame or m.edge_cartesian
                or m.use_flash_atten or any(v is not None for v in
                    (m.attn_v_proj,m.attn_o_proj,m.attn_focus_mix,m.node_cartesian_tp))):
                raise ValueError("unsupported SO(2) attention contract")
            if variant=="reverse":
                bind(m,_transport_forward(m,calls,native_forward=True))
            if variant=="packed":
                for layer in m.so2_linears:
                    caches.append((layer,layer._cached_weight))
                    with torch.no_grad():
                        layer._cached_weight=layer._build_so2_weight().detach()
            if variant in ("edge","combined","full","transport","local"):
                mixer=m.radial_degree_mixer
                if mixer is None or mixer.rank!=1 or mixer.channels!=32 or mixer.reduced_dim!=7:
                    raise ValueError("requires rank-one degree-channel mixer")
                bind(mixer,mix(mixer))
                bind(m,_transport_forward(m,calls) if variant in ("transport","local") else _edge_forward(m,calls))
            if variant in ("linear","combined","full","transport","local"):
                for layer in m.so2_linears:
                    bind(layer,lin(layer))
            if variant in ("full","transport","local"):
                from deepmd.pt.model.descriptor.sezm_nn.activation import GatedActivation
                for activation in m.non_linearities:
                    if isinstance(activation,GatedActivation):
                        bind(activation,gate(activation))
            if variant=="local":
                message_edits.append((m,m.__dict__.get("so2_message")))
                m.so2_message=_local_message(m,calls)
        yield calls
    finally:
        for m,old in reversed(message_edits):
            if old is None:
                del m.so2_message
            else:
                m.so2_message=old
        for layer,old in caches:
            layer._cached_weight=old
        for m,old in reversed(edits):
            if old is None:
                del m.forward
            else:
                m.forward=old
