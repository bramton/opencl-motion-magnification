#define EPS 0x1.0p-23f // TODO: defined already as FLT_EPSILON somewhere?

__kernel void pdaa_kernel( __global const uchar* _lap, int lap_step, int lap_offset, int rows, int cols,
                          __global const uchar* _lap_prev,
                          __global const uchar* _riesz,  int riesz_step, int riesz_offset,                                                            
                          __global const uchar* _riesz_prev,
                          __global uchar* _amp,
                          __global uchar* _pd // Phase diff
                          )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x >= cols || y >= rows) { return;}

    __global const float  *lap = ((__global const float*)( _lap + mad24(y, lap_step, lap_offset)) + x); 
    __global const float  *lap_prev = ((__global const float*)( _lap_prev + mad24(y, lap_step, lap_offset)) + x);

    __global const float2 *riesz = ((__global const float2*)( _riesz + mad24(y, riesz_step, riesz_offset)) + x);
    __global const float2 *riesz_prev = ((__global const float2*)( _riesz_prev + mad24(y, riesz_step, riesz_offset)) + x);

    __global float        *amp = ((__global float*)( _amp + mad24(y, lap_step, lap_offset)) + x);
    __global float2       *phase_diff = ((__global float2*)( _pd + mad24(y, riesz_step, riesz_offset)) + x);

    float  q_conj_prod_real, q_conj_prod_amp, phase_diff_v;
    float2 q_conj_prod, orientation;

    // TODO: optimise using vector operations
    q_conj_prod_real = *lap * *lap_prev + (*riesz).x * (*riesz_prev).x + (*riesz).y * (*riesz_prev).y;
    q_conj_prod = mad(-(*lap), *riesz_prev, *lap_prev * *riesz);

    q_conj_prod_amp = length((float4)(q_conj_prod_real, q_conj_prod, 0.0f));
    phase_diff_v = acos(q_conj_prod_real/(EPS + q_conj_prod_amp));

    orientation = q_conj_prod/(EPS + length(q_conj_prod));

    *phase_diff = phase_diff_v * orientation;
    *amp = sqrt(q_conj_prod_amp);
}