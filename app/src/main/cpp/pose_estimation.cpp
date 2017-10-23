//
// Created by betty on 2017/9/5.
//
#include "pose_estimation.h"
#include "jni.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv/cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/video/video.hpp"
#include "Eigen/Dense"
#include <Eigen/Eigen>
#include <iostream>
#include <fstream>
#include "android/log.h"

//#define LOG_TAG "FastMcd/opencv-lib"
//#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO,LOG_TAG,__VA_ARGS__))

using namespace std;
using namespace cv;
using namespace Eigen;
extern "C"
{
Vector3d pre_p_wi;
Vector3d pre_v_wi;
Quaterniond pre_q_wi;
Vector3d pre_w_m;
Vector3d pre_a_m;
double delta_t;
Vector3d pre_b_w;
Vector3d pre_b_a;
Vector3d pre_n_w;
Vector3d pre_n_a;
Vector3d pre_n_bw;
Vector3d pre_n_ba;
double L;
Matrix3d pre_R_wi;
Vector3d g(0, 0, 9.8);

Vector3d evaluate_p;
Vector3d evaluate_v;
Quaterniond evaluate_q;
Vector3d evaluate_theta;
Vector3d evaluate_b_w;
Vector3d evaluate_b_a;
double evaluate_L = 1;

Vector3d residual_p_t;
Vector3d residual_v_t;
Quaterniond residual_q_t;
Vector3d residual_theta;
Vector3d residual_b_w_t;
Vector3d residual_b_a_t;
double residual_L_t = 1;

Vector3d residual_p_t_1;
Vector3d residual_v_t_1;
Quaterniond residual_q_t_1;
Vector3d residual_theta_t_1;
Vector3d residual_b_w_t_1;
Vector3d residual_b_a_t_1;
double residual_L_1 = 1;

Vector3d cur_p_wi;
Vector3d cur_v_wi;
Quaterniond cur_q_wi;
Vector3d cur_w_m;
Vector3d cur_a_m;
Vector3d cur_b_w;
Vector3d cur_b_a;

Vector3d correction_p;
Vector3d correction_v;
Quaterniond correction_q;
Vector3d correction_b_w;
Vector3d correction_b_a;
double correction_L;

Matrix<double, 6, 6> Rm;
Matrix<double, 16, 16> P_k;
Matrix<double, 16, 16> P_k1;
Matrix<double, 16, 16> P_k1k1;

Vector3d cur_z_p;
Quaterniond cur_z_q;
Matrix3d cur_z_R;

Vector3d pre_z_p;
Quaterniond pre_z_q;
Matrix3d pre_z_R;

Vector3d fused_p;
Vector3d fused_v;
Quaterniond fused_q;
Vector3d fused_b_w;
Vector3d fused_b_a;
double fused_L;

JNIEXPORT jdoubleArray JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_camMatrixFromJNI(JNIEnv *env, jclass type,
                                                              jlong nativeObjAddr,
                                                              jlong nativeObjAddr1) {
    // TODO
    vector<cv::KeyPoint> keypoints_1, keypoints_2;
    vector<cv::Point2f> points_1;
    vector<cv::Point2f> points_2;
    cv::Mat R, t;
    cv::Mat &current_frame = *(cv::Mat *) nativeObjAddr;
    cv::Mat &first_frame = *(cv::Mat *) nativeObjAddr1;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    unsigned int maxCout = 300;//定义最大个数
    double minDis = 10;//定义最小距离
    double qLevel = 0.01;//定义质量水平
    //压缩图片
    Mat tmpImg, tmpImg_first;
    Size sz;
    pyrDown(first_frame, tmpImg_first, sz, BORDER_DEFAULT);
    pyrDown(tmpImg_first, tmpImg_first, sz, BORDER_DEFAULT);
    pyrDown(current_frame, tmpImg, sz, BORDER_DEFAULT);
    pyrDown(tmpImg, tmpImg, sz, BORDER_DEFAULT);

    //Fast提取角点
//    FAST(tmpImg_first, keypoints_1, fast_threshold, nonmaxSuppression);
//    KeyPoint:: convert(keypoints_1, points_1, vector<int>());

    goodFeaturesToTrack(tmpImg_first, points_1, maxCout, qLevel, minDis);
    vector<uchar> status;
    vector<float> err;
    status.reserve(maxCout);
    err.reserve(maxCout);
    Size winSize = Size(13, 13);
    TermCriteria termcrit = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
//    Mat gray_first, gray_current;
//    first_frame.convertTo(gray_first, CV_8U);
//    current_frame.convertTo(gray_current, CV_8U);
// getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    if (points_1.empty()) {
        return 0;
    }
    calcOpticalFlowPyrLK(tmpImg_first, tmpImg, points_1, points_2, status, err, winSize, 3, termcrit, 0, 0.001);
    unsigned int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        Point2f pt = points_2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points_1.erase(points_1.begin() + (i - indexCorrection));
            points_2.erase(points_2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
    double K[3][3] = {412.1066, 0, 188.7078, 0, 413.9122, 259.5036, 0, 0, 1};
    for (int i = 0; i < points_2.size(); i++) {
        circle(tmpImg, points_2.at(i), 1, Scalar(10, 10, 255), 1, 8, 0);
    }
    //计算本质矩阵可以自行SVD加快处理速度
    cv::Point2d principal_point(K[0][2], K[1][2]);//相机光心
    double focal_length = 0.5 * (K[0][0] + K[1][1]);//相机焦距
    cv::Mat essential_matrix, mask;
    if (points_2.empty()) {
        return 0;
    }
    essential_matrix = findEssentialMat(points_1, points_2, focal_length, principal_point, RANSAC, 0.999, 1.0, mask);
    if (essential_matrix.cols != 3 || essential_matrix.rows != 3) {
        return 0;
    }
    //从本质矩阵恢复旋转和平移信息 t是默认归一化的矩阵
    recoverPose(essential_matrix, points_1, points_2, R, t, focal_length, principal_point, mask);
    if ((!R.data) && (!t.data)) {
        return NULL;
    }
    jdoubleArray resultarray = env->NewDoubleArray((R.rows + 1) * R.cols);
    jdouble *element;
    element = env->GetDoubleArrayElements(resultarray, JNI_FALSE);
    for (int i = 0; i < R.rows; i++) {
        for (int j = 0; j < R.cols; j++) {
            element[i * R.cols + j] = R.at<double>(i, j);
        }
    }
    for (int m = 0; m < t.rows; m++) {
        for (int n = 0; n < t.cols; n++) {
            element[9 + (m * t.cols + n)] = t.at<double>(m, n);
        }
    }
    pyrUp(tmpImg, tmpImg, sz, BORDER_DEFAULT);
    pyrUp(tmpImg, current_frame, sz, BORDER_DEFAULT);
    pyrUp(tmpImg_first, tmpImg_first, sz, BORDER_DEFAULT);
    pyrUp(tmpImg_first, first_frame, sz, BORDER_DEFAULT);
    env->ReleaseDoubleArrayElements(resultarray, element, 0);
    return resultarray;
}

JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_evaluateIMU(JNIEnv *env, jclass type,
                                                         jfloatArray p_wi_,
                                                         jfloatArray v_wi_, jfloatArray q_wi_,
                                                         jfloatArray w_, jfloatArray a_,
                                                         jfloat dT) {
    jfloat *p_wi = env->GetFloatArrayElements(p_wi_, NULL);
    jfloat *v_wi = env->GetFloatArrayElements(v_wi_, NULL);
    jfloat *q_wi = env->GetFloatArrayElements(q_wi_, NULL);
    jfloat *w = env->GetFloatArrayElements(w_, NULL);
    jfloat *a = env->GetFloatArrayElements(a_, NULL);

    // TODO
    pre_p_wi << p_wi[0], pre_p_wi[1], pre_p_wi[2];

    pre_v_wi << v_wi[0], v_wi[1], v_wi[2];

    pre_q_wi = Quaternion<double>(q_wi[0], q_wi[1], q_wi[2], q_wi[3]);

    pre_q_wi.normalize();
    pre_w_m << w[0], w[1], w[2];
    pre_a_m << a[0], a[1], a[2];
    delta_t = dT;
    pre_b_w << 0.053, 0.053, 0.053;
    pre_b_a << 0.245, 0.245, 0.245;
    pre_n_w << 1.19e-4, 1.19e-4, 1.19e-4;
    pre_n_a << 1.77e-3, 1.77e-3, 1.77e-3;
    pre_n_bw << 2.74e-3, 2.74e-3, 2.74e-3;
    pre_n_ba << 0.06, 0.06, 0.06;
    pre_n_w << 1.42e-8, 1.42e-8, 1.42e-8;
    pre_n_a << 3.11e-6, 3.11e-6, 3.11e-6;
    L = 1.0;
    Matrix<double, 3, 1> residual_w;
    Matrix<double, 3, 1> residual_a;
    Matrix3d skew_w;
    residual_w = pre_w_m - pre_b_w;
    residual_a = pre_a_m - pre_b_a;
    skew_w << 0, -residual_w[2], residual_w[1],
            residual_w[2], 0, -residual_w[0],
            -residual_w[1], residual_w[0], 0;

    Matrix3d skew_a;
    skew_a << 0, -residual_a[2], residual_a[1],
            residual_a[2], 0, -residual_a[0],
            -residual_a[1], residual_a[0], 0;
    pre_R_wi = pre_q_wi.toRotationMatrix();

    //IMU先验状态预测
    evaluate_p = pre_p_wi + pre_v_wi * delta_t;
    evaluate_v = pre_v_wi + (pre_R_wi * (pre_a_m - pre_b_a) - g) * delta_t;
    evaluate_q = pre_q_wi * Quaterniond(1, 1 / 2 * (pre_w_m[0] - pre_b_w[0]) * delta_t,
                                        1 / 2 * (pre_w_m[1] - pre_b_w[1]) * delta_t,
                                        1 / 2 * (pre_w_m[2] - pre_b_w[2]) * delta_t);
    evaluate_b_w = pre_b_w;
    evaluate_b_a = pre_b_a;
    evaluate_L = L;

    residual_b_a_t_1.setZero();
    residual_b_w_t_1.setZero();
    residual_q_t.setIdentity();
    residual_theta_t_1.setZero();
    residual_p_t_1.setZero();
    residual_v_t_1.setZero();

    //IMU误差传播方程
//    residual_L_t = residual_L_1 +
    residual_b_a_t = residual_b_a_t_1 + pre_n_ba * delta_t;
    residual_b_w_t = residual_b_w_t_1 + pre_n_bw * delta_t;
//    residual_q_t = residual_q_t_1 - skew_w * residual_q_t_1 * delta_t - residual_b_w_t_1 * delta_t - pre_n_w * delta_t;
    residual_theta = residual_theta_t_1 - skew_w * residual_theta_t_1 * delta_t -
                     residual_b_w_t_1 * delta_t - pre_n_w * delta_t;
    residual_v_t = residual_v_t_1 - pre_R_wi * skew_a * residual_theta_t_1 * delta_t -
                   pre_R_wi * residual_b_a_t_1 * delta_t - pre_R_wi * pre_n_a * delta_t;
    residual_p_t = residual_p_t_1 + residual_b_w_t_1 * delta_t;

    residual_b_a_t_1 = residual_b_a_t;
    residual_b_w_t_1 = residual_b_w_t;
//    residual_q_t_1 = residual_q_t;
    residual_theta_t_1 = residual_theta;
    residual_v_t_1 = residual_v_t;
    residual_p_t_1 = residual_p_t;

    env->ReleaseFloatArrayElements(p_wi_, p_wi, 0);
    env->ReleaseFloatArrayElements(v_wi_, v_wi, 0);
    env->ReleaseFloatArrayElements(q_wi_, q_wi, 0);
    env->ReleaseFloatArrayElements(w_, w, 0);
    env->ReleaseFloatArrayElements(a_, a, 0);
}
JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_imuCoreFromJNI(JNIEnv *env, jclass type,
                                                            jfloatArray p_wi_, jfloatArray v_wi_,
                                                            jfloatArray q_wi_, jfloatArray w_,
                                                            jfloatArray a_, jfloat dT) {
    jfloat *p_wi = env->GetFloatArrayElements(p_wi_, NULL);
    jfloat *v_wi = env->GetFloatArrayElements(v_wi_, NULL);
    jfloat *q_wi = env->GetFloatArrayElements(q_wi_, NULL);
    jfloat *w = env->GetFloatArrayElements(w_, NULL);
    jfloat *a = env->GetFloatArrayElements(a_, NULL);

    // TODO
    cur_p_wi << p_wi[0], p_wi[1], p_wi[2];
    cur_v_wi << v_wi[0], v_wi[1], v_wi[2];
    cur_q_wi = Quaterniond(q_wi[0], q_wi[1], q_wi[2], q_wi[3]);
    cur_q_wi.normalize();
    cur_w_m << w[0], w[1], w[2];
    cur_a_m << a[0], a[1], a[2];
    cur_b_w << 0.053, 0.053, 0.053;
    cur_b_a << 0.245, 0.245, 0.245;
    L = 1.0;
    double delta_t = dT;
    //propagateState
    Vector3d dv;
    Vector3d ew;
    ew = cur_w_m - cur_b_w;
    Vector3d ewold;
    ewold = pre_w_m - pre_b_w;
    Vector3d ea;
    ea = cur_a_m - cur_b_a;
    Vector3d eaold;
    eaold = pre_a_m = pre_b_a;
    Matrix4d Omega;
    Omega << 0, ea[2], -ea[1], ea[0],
            -ea[2], 0, ea[0], ea[1],
            ea[1], -ea[0], 0, ea[2],
            -ea[0], -ea[1], -ea[2], 0;
    Matrix4d OmegaOld;
    OmegaOld << 0, eaold[2], -eaold[1], eaold[0],
            -eaold[2], 0, eaold[0], eaold[1],
            eaold[1], -eaold[0], 0, eaold[2],
            -eaold[0], -eaold[1], -eaold[2], 0;
    Matrix4d OmegaMean;
    Vector3d mean;
    mean = (ew + ewold) / 2.0;
    OmegaMean << 0.0, mean[2], -mean[1], mean[0],
            -mean[2], 0, mean[0], mean[1],
            mean[1], -mean[0], 0, mean[2],
            -mean[0], -mean[1], -mean[2], 0;
    int div = 1;
    Matrix4d MatExp;
    MatExp.setIdentity();
    OmegaMean *= 0.5 * delta_t;
    for (int i = 1; i < 5; i++) {
        div *= i;
        MatExp = MatExp + OmegaMean / div;
        OmegaMean *= OmegaMean;
    }
    // first oder quat integration matrix
    Matrix4d quat_int =
            MatExp + 1.0 / 48.0 * (Omega * OmegaOld - OmegaOld * Omega) * delta_t * delta_t;
    cur_q_wi.coeffs() = quat_int * pre_q_wi.coeffs();
    cur_q_wi.normalize();
    Matrix3d cur_R_wi;
    cur_R_wi = cur_q_wi.toRotationMatrix();
    Matrix3d pre_R_wi;
    pre_R_wi = pre_q_wi.toRotationMatrix();
    dv = (cur_R_wi * ea + pre_R_wi * eaold) / 2.0;
    cur_v_wi = pre_v_wi + dv * delta_t;
    cur_p_wi = pre_p_wi + ((cur_v_wi + pre_v_wi) / 2.0) * delta_t;

    //noise
    Matrix<double, 3, 1> nav;
    nav << 3.11e-6, 3.11e-6, 3.11e-6;
    Matrix<double, 3, 1> nbav;
    nbav << 6e-2, 6e-2, 6e-2;
    Matrix<double, 3, 1> nwv;
    nwv << 1.42e-8, 1.42e-8, 1.42e-8;
    Matrix<double, 3, 1> nbwv;
    nbwv << 2.74e-3, 2.74e-3, 2.74e-3;

    Matrix3d a_sk;
    a_sk << 0.0, -ea[2], ea[1],
            ea[2], 0.0, -ea[0],
            -ea[1], ea[0], 0.0;
    Matrix3d w_sk;
    w_sk << 0.0, -ew[2], ew[1],
            ew[2], 0.0, -ew[0],
            -ew[1], ew[0], 0.0;
    //predictProcessCovariance--Fd
    Matrix3d I3;
    I3.setIdentity();
    const double dt_p2_2 = delta_t * delta_t * 0.5; // dt^2 / 2
    const double dt_p3_6 = dt_p2_2 * delta_t / 3.0; // dt^3 / 6
    const double dt_p4_24 = dt_p3_6 * delta_t * 0.25; // dt^4 / 24
    const double dt_p5_120 = dt_p4_24 * delta_t * 0.2; // dt^5 / 120
    Matrix3d Ca3, A, B, C, D, E, F;
    Ca3 = cur_R_wi * a_sk;
    A = Ca3 * (-dt_p2_2 * I3 + dt_p3_6 * w_sk - dt_p4_24 * w_sk * w_sk);
    B = Ca3 * (dt_p3_6 * I3 - dt_p4_24 * w_sk + dt_p5_120 * w_sk * w_sk);
    D = -A;
    E = I3 - delta_t * w_sk + dt_p2_2 * w_sk * w_sk;
    F = -delta_t * I3 + dt_p2_2 * w_sk - dt_p3_6 * w_sk * w_sk;
    C = Ca3 * F;
    Matrix<double, 16, 16> Fd;
    Fd.setIdentity();
    Fd.block(0, 3, 3, 3) = I3 * delta_t;
    Fd.block(0, 6, 3, 3) = A;
    Fd.block(0, 9, 3, 3) = B;
    Fd.block(0, 12, 3, 3) = -cur_R_wi * dt_p2_2;
    Fd.block(3, 9, 3, 3) = C;
    Fd.block(3, 9, 3, 3) = D;
    Fd.block(3, 12, 3, 3) = -cur_R_wi * delta_t;
    Fd.block(6, 6, 3, 3) = E;
    Fd.block(6, 9, 3, 3) = F;
    //noiseMatrix---Gc
    Matrix<double, 16, 12> Gc;
    Gc.setZero();
    Gc.block(3, 0, 3, 3) = -cur_R_wi;
    Gc.block(6, 6, 3, 3) = -I3;
    Gc.block(9, 9, 3, 3) = I3;
    Gc.block(12, 3, 3, 3) = I3;
    //Matrix---Qc
    Matrix<double, 12, 12> Qc;
    Qc.setIdentity();
    Qc.block(0, 0, 3, 3) = 3.11e-6 * I3;
    Qc.block(3, 3, 3, 3) = 6e-2 * I3;
    Qc.block(6, 6, 3, 3) = 1.42e-8 * I3;
    Qc.block(9, 9, 3, 3) = 2.74e-3 * I3;
    //Matric---Qd
    Matrix<double, 16, 16> Qd;
    Qd = Gc * Qc * Gc.transpose() * delta_t;
    //预测误差协方差矩阵
    P_k.setIdentity();
    P_k1 = Fd * P_k * Fd.transpose() + Qd;

    env->ReleaseFloatArrayElements(p_wi_, p_wi, 0);
    env->ReleaseFloatArrayElements(v_wi_, v_wi, 0);
    env->ReleaseFloatArrayElements(q_wi_, q_wi, 0);
    env->ReleaseFloatArrayElements(w_, w, 0);
    env->ReleaseFloatArrayElements(a_, a, 0);
}
JNIEXPORT void JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_camCoreFromJNI(JNIEnv *env, jclass type,
                                                            jfloatArray z_p_, jfloatArray z_q_,
                                                            jfloatArray pre_w_p_,
                                                            jfloatArray pre_w_q_) {
    jfloat *z_p = env->GetFloatArrayElements(z_p_, NULL);
    jfloat *z_q = env->GetFloatArrayElements(z_q_, NULL);
    jfloat *pre_w_p = env->GetFloatArrayElements(pre_w_p_, NULL);
    jfloat *pre_w_q = env->GetFloatArrayElements(pre_w_q_, NULL);

//    ofstream fout("value.txt");
    // TODO
    const double n_zp = 0.01;
    const double n_zq = 0.02;
    const double s_zp = n_zp * n_zp;
    const double s_zq = n_zq * n_zq;
    cur_z_p << z_p[0], z_p[1], z_p[2];
    cur_z_q = Quaterniond(z_q[0], z_q[1], z_q[2], z_q[3]);
    cur_z_R = cur_z_q.toRotationMatrix();

//    fout << "每幅图像的位置坐标：" << endl << endl;
//    fout << "z_p =  "<<z_p[0]<<"   "<<z_p[1]<<"   "<<z_p[2]<<endl ;
//    fout << "每幅图像的四元数：" << endl << endl;
//    fout << "z_q =  "<<z_q[0]<<"   "<<z_q[1]<<"   "<<z_q[2]<<"   "<<z_q[3]<<endl ;
//
//    fout << "IMU的位置坐标：" << endl << endl;
//    fout << "imu_p =  "<<cur_p_wi[0]<<"   "<<cur_p_wi[1]<<"   "<<cur_p_wi[2]<<endl ;
//    fout << "IMU的四元数：" << endl << endl;
//    fout << "imu_q =  "<<cur_q_wi.w()<<"   "<<cur_q_wi.x()<<"   "<<cur_q_wi.y()<<"   "<<cur_q_wi.z()<<endl ;

    pre_z_p << pre_w_p[0], pre_w_p[1], pre_w_p[2];
    pre_z_q = Quaterniond(pre_w_q[0], pre_w_q[1], pre_w_q[2], pre_w_q[3]);
    pre_z_R = pre_z_q.toRotationMatrix();

    Rm.setIdentity();
    Rm(0, 0) = s_zp;
    Rm(1, 1) = s_zp;
    Rm(2, 2) = s_zp;
    Rm(3, 3) = s_zq;
    Rm(4, 4) = s_zq;
    Rm(5, 5) = s_zq;

    // H matrix
    Matrix<double, 6, 16> H;
    H.setZero();
    Matrix3d I;
    I.setIdentity();
    H.block(0, 0, 3, 3) = L * I;
    H.block(0, 15, 3, 1) = cur_p_wi;
    H.block(3, 6, 3, 3) = cur_z_R.transpose();
    //camera residuals
    Matrix<double, 6, 1> residual_cam;




    // residuals of position
    //  r_old.block<3, 1> (0, 0) = z_p_ - C_wv.transpose() * (state_old.p_ + C_q.transpose() * state_old.p_ci_) * state_old.L_;
    residual_cam.block(0, 0, 3, 1) = cur_z_p - cur_z_R.transpose() * pre_z_p * L;
    Quaterniond q_err;
    q_err = ((pre_q_wi * pre_z_q).conjugate()) * cur_z_q;
    // residuals of attitude
    residual_cam.block(3, 0, 3, 1) = q_err.vec() / q_err.w() * 2;


    Matrix<double, 6, 6> S;
    Matrix<double, 16, 6> Kk;
    Matrix<double, 16, 16> Id;
    Matrix<double, 16, 1> correction_;
    S = H * P_k1 * H.transpose() + Rm;
    Kk = P_k1 * H.transpose() * S.inverse();

    correction_ = Kk * residual_cam;
    const MatrixXd KH = (Id - Kk * H);
    P_k1k1 = KH * P_k1 * KH.transpose() + Kk * Rm * Kk.transpose();
    // make sure P stays symmetric
    P_k1k1 = 0.5 * (P_k1k1 + P_k1k1.transpose());
    P_k = P_k1k1;

    correction_p = residual_p_t_1 + correction_.block(0, 0, 3, 1);
    correction_v = residual_v_t_1 + correction_.block(3, 0, 3, 1);
    correction_b_w = residual_b_w_t_1 + correction_.block(9, 0, 3, 1);
    correction_b_a = residual_b_a_t_1 + correction_.block(12, 0, 3, 1);
    correction_L = residual_L_1 + correction_(15);

    fused_p = evaluate_p + correction_p;
    fused_v = evaluate_v + correction_v;
    fused_b_w = evaluate_b_w + correction_b_w;
    fused_b_a = evaluate_b_a + correction_b_a;
    fused_L = evaluate_L + correction_L;

    //姿态四元数
    double theta_x, theta_y, theta_z;
    theta_x = correction_(6);
    theta_y = correction_(7);
    theta_z = correction_(8);

    double sum_theta = theta_x * theta_x + theta_y * theta_y + theta_z * theta_z;
    if ((sum_theta / 4) <= 1) {
        correction_q.w() = sqrt(1 - sum_theta / 4);
        correction_q.x() = theta_x / 2;
        correction_q.y() = theta_y / 2;
        correction_q.z() = theta_z / 2;
    } else {
        correction_q.w() = 1 / sqrt(1 + sum_theta / 4);
        correction_q.x() = (1 / sqrt(1 + sum_theta / 4)) * theta_x / 2;
        correction_q.y() = (1 / sqrt(1 + sum_theta / 4)) * theta_y / 2;
        correction_q.z() = (1 / sqrt(1 + sum_theta / 4)) * theta_z / 2;
    }
    fused_q = correction_q * evaluate_q;

//    fout << "IMU的位置坐标：" << endl << endl;
//    fout << "final_p =  "<<final_p[0]<<"   "<<final_p[1]<<"   "<<final_p[2]<<endl ;
//    fout << "IMU的四元数：" << endl << endl;
//    fout << "final_q =  "<<final_q.w()<<"   "<<final_q.x()<<"   "<<final_q.y()<<"   "<<final_q.z()<<endl ;

    env->ReleaseFloatArrayElements(z_p_, z_p, 0);
    env->ReleaseFloatArrayElements(z_q_, z_q, 0);
    env->ReleaseFloatArrayElements(pre_w_p_, pre_w_p, 0);
    env->ReleaseFloatArrayElements(pre_w_q_, pre_w_q, 0);
}
JNIEXPORT jdoubleArray JNICALL
Java_com_example_betty_imucam_1klt_NativeFun_FusedData(JNIEnv *env, jclass type) {

    // TODO
    jdoubleArray fused_pq = env->NewDoubleArray(7 * 1);
    jdouble *element;
    element = env->GetDoubleArrayElements(fused_pq, JNI_FALSE);
    element[0] = fused_p(0, 0);
    element[1] = fused_p(1, 0);
    element[2] = fused_p(2, 0);
    element[3] = fused_q.w();
    element[4] = fused_q.x();
    element[5] = fused_q.y();
    element[6] = fused_q.z();
    env->ReleaseDoubleArrayElements(fused_pq, element, 0);
    return fused_pq;
}
}


