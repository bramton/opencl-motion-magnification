#define DIG(a) a,
__constant float w[] = { FILT_WEIGHTS };

__kernel void buttw_kernel(__global const uchar* _phase, int phase_step, int phase_offset, int phase_rows, int phase_cols, 
                           __global uchar* _state, int state_step, int state_offset,
                           __global uchar* _dst                                                                
                          )
{

    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x >= phase_cols || y >= phase_rows) { return;}

    __global const float2 *phase = ((__global const float2*)( _phase + mad24(y, phase_step, phase_offset)) + x);
    __global float2       *dst   = ((__global float2*)      ( _dst  +  mad24(y, phase_step, phase_offset)) + x);
    __global float4       *state = ((__global float4*)      ( _state + mad24(y, state_step, state_offset)) + mul24(x, NBIQUADS));

    float2 out, in;
    float k;
    // b0 = 1, b1 = 0, b2 = -1, a0 = 1
    in = *phase;
    for (int i = 0; i < NBIQUADS; i++) {
        // direct implementation of transposed direct form 2
        k = w[i*3 + 0]; // gain
        out = k*in + state[i].xy;
        state[i].xy = state[i].zw + 0.0f*in - w[i*3 + 1]*out;
        state[i].zw = -k*in - w[i*3 + 2]*out;
        in = out;
    }
    
    // https://en.wikipedia.org/wiki/Digital_biquad_filter
    *dst = out;
}