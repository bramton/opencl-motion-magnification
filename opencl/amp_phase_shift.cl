#define EPS 0x1.0p-23f // TODO: defined already as FLT_EPSILON somewhere?

__kernel void aps_kernel( __global const uchar* _lap, int lap_step, int lap_offset, int rows, int cols,
                          __global const uchar* _riesz,  int riesz_step, int riesz_offset,                                                            
                          __global const uchar* _phase,
                          __global uchar* _lap_amp
                          )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if( x >= cols || y >= rows) { return;}

    __global const float *lap = ((__global const float*)( _lap + mad24(y, lap_step, lap_offset)) + x); 

    __global const float2 *riesz = ((__global const float2*)( _riesz + mad24(y, riesz_step, riesz_offset)) + x);
    __global const float2 *phase = ((__global const float2*)( _phase + mad24(y, riesz_step, riesz_offset)) + x);

    __global float* lap_amp = ((__global float*)( _lap_amp + mad24(y, lap_step, lap_offset)) + x);

    float phase_mag, exp_phase_real;
    float2 phase_amp, exp_phase;

    phase_amp = *phase * ALPHA;
    phase_mag = EPS + length(phase_amp);
    exp_phase_real = cos(phase_mag);

    exp_phase = phase_amp/phase_mag * sin(phase_mag); // TODO: correct??

    *lap_amp = exp_phase_real * *lap - exp_phase.x * (*riesz).x - exp_phase.y * (*riesz).y;
}