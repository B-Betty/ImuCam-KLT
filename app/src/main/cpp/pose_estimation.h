//
// Created by betty on 2017/9/5.
//

#include "jni.h"

#ifndef IMUCAMTEST_POSE_ESTIMATION_H
#define IMUCAMTEST_POSE_ESTIMATION_H

#ifdef __cplusplus
extern "C" {
#endif

JNIEXPORT jdoubleArray JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_camMatrixFromJNI(JNIEnv *env, jclass type,
                                                              jlong nativeObjAddr,
                                                              jlong nativeObjAddr1);

JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_evaluateIMU(JNIEnv *env, jclass type,
                                                         jfloatArray p_wi_, jfloatArray v_wi_,
                                                         jfloatArray q_wi_, jfloatArray w_,
                                                         jfloatArray a_, jfloat dT) ;

JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_imuCoreFromJNI(JNIEnv *env, jclass type,
                                                            jfloatArray p_wi_, jfloatArray v_wi_,
                                                            jfloatArray q_wi_, jfloatArray w_,
                                                            jfloatArray a_, jfloat dT) ;

JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_camCoreFromJNI(JNIEnv *env, jclass type,
                                                            jfloatArray z_p_, jfloatArray z_q_,
                                                            jfloatArray pre_w_p_,
                                                            jfloatArray pre_w_q_) ;

JNIEXPORT jdoubleArray JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_FusedData(JNIEnv *env, jclass type) ;

#ifdef __cplusplus
}
#endif
#endif
