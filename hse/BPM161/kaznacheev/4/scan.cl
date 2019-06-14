__kernel void scan(__global double * a, __global double * r, __global double * t, __local double * b, int N)
{
    uint glb = get_global_id(0);
    uint lcl = get_local_id(0);
    uint lcl_size = get_local_size(0);
    uint group = get_group_id(0);

    if (glb < N) {
        uint dp = 1;
        b[lcl] = a[glb];
        for(uint s = lcl_size>>1; s > 0; s >>= 1) {
            barrier(CLK_LOCAL_MEM_FENCE);
            if(lcl < s) {
                uint i = dp*(2*lcl+1)-1;
                uint j = dp*(2*lcl+2)-1;
                b[j] += b[i];
            }

            dp <<= 1;
        }

        if(lcl == 0) {
            b[lcl_size - 1] = 0;
        }

        for(uint s = 1; s < lcl_size; s <<= 1) {
            dp >>= 1;
            barrier(CLK_LOCAL_MEM_FENCE);

            if(lcl < s) {
                uint i = dp*(2*lcl+1)-1;
                uint j = dp*(2*lcl+2)-1;

                int t = b[j];
                b[j] += b[i];
                b[i] = t;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (lcl == lcl_size - 1) {
            t[group] = b[lcl] + a[glb];
        }
        r[glb] = b[lcl];
    }
}

 __kernel void add(__global double * r, __global double * t, int N) {
    int glb = get_global_id(0);
    int group = get_group_id(0);
    if (glb < N) {
        r[glb] += t[group];
    }
}