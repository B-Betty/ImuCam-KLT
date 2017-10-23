package com.example.betty.imucam_klt;

/**
 * Created by betty on 2017/9/5.
 */

public class NativeFun {
    public static native String stringFromJNI();
    public static native double[] camMatrixFromJNI(long nativeObjAddr, long nativeObjAddr1);
    public static native void evaluateIMU(float[] p_wi,float[] v_wi,float[] q_wi,float[] w,float[] a,float dT);
    public static native void imuCoreFromJNI(float[] p_wi,float[] v_wi,float[] q_wi,float[] w,float[] a,float dT);
    public static native void camCoreFromJNI(float[] z_p,float[] z_q,float[] pre_w_p,float[] pre_w_q);
    public static native double[] FusedData();
}
