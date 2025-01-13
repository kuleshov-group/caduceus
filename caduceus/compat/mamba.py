from importlib.metadata import PackageNotFoundError, version as pkg_version
from packaging.version import parse, Version

def get_mamba_version() -> Version | None:
    try:
        
        return parse(pkg_version('mamba-ssm'))
    except PackageNotFoundError:
        return None


def get_mamba_modules():
    """Get mamba normalization modules based on available version.
    
    Returns:
        tuple: (Block, RMSNorm, layer_norm_fn, rms_norm_fn)
    """
    version = get_mamba_version()
    if version is None:
        return None, None, None, None
    if version.major < 2:
        from mamba_ssm.modules.mamba_simple import Block
        from mamba_ssm.ops.triton.layernorm import (  # v1 structure
            RMSNorm, 
            layer_norm_fn, 
            rms_norm_fn
        )
        return Block, RMSNorm, layer_norm_fn, rms_norm_fn
    else:
        from mamba_ssm.modules.block import Block  
        from mamba_ssm.ops.triton.layer_norm import (  # v2 structure
            RMSNorm, 
            layer_norm_fn, 
            rms_norm_fn
        )
        return Block, RMSNorm, layer_norm_fn, rms_norm_fn

Block, RMSNorm, layer_norm_fn, rms_norm_fn = get_mamba_modules()