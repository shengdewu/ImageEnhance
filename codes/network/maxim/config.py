maxim_config = {
    # params: 6.108515000000001 M, GFLOPS: 93.163716608
    "MAXIM_S-1": {
        "features": 32,
        "depth": 3,
        "num_stages": 1,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
    # params: 13.35383 M, GFLOPS: 206.743273472
    "MAXIM_S-2": {
        "features": 32,
        "depth": 3,
        "num_stages": 2,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
    # params: 20.599145 M, GFLOPS: 320.32194560000005
    "MAXIM_S-3": {
        "features": 32,
        "depth": 3,
        "num_stages": 3,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
    # params: 19.361219000000002 M, 308.495712256 GFLOPs
    "MAXIM_M-1": {
        "features": 64,
        "depth": 3,
        "num_stages": 1,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
    # params: 40.83911 M, 675.25541888 GFLOPs
    "MAXIM_M-2": {
        "features": 64,
        "depth": 3,
        "num_stages": 2,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
    # params: 62.317001 M, 1042.014666752 GFLOPs
    "MAXIM_M-3": {
        "features": 64,
        "depth": 3,
        "num_stages": 3,
        "num_groups": 2,
        "num_bottleneck_blocks": 2,
        "block_gmlp_factor": 2,
        "grid_gmlp_factor": 2,
        "input_proj_factor": 2,
        "channels_reduction": 4,
    },
}
