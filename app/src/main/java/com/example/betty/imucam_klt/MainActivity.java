package com.example.betty.imucam_klt;

import android.content.pm.ActivityInfo;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Build;
import android.os.Environment;
import android.os.Handler;
import android.support.annotation.RequiresApi;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity implements SensorEventListener, CameraBridgeViewBase.CvCameraViewListener2{

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
        System.loadLibrary("pose_estimation");
        System.loadLibrary("opencv_java3");
    }
    //visual
    Mat mRgb1;
    Mat mGray;
    Mat first_frame;
    private CameraBridgeViewBase cameraBridgeViewBase;
    final String TAG = "pose-estimation";
    private int numFrame = 0;
    float[] p_wc = new float[3];
    float[] q_wc = new float[4];
    float[] R_wc = new float[9];
    float[] pre_p_wc = new float[3];
    float[] pre_q_wc = new float[4];
    float[] Vector_wc = new float[3];
    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case BaseLoaderCallback.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    //开启摄像头
                    cameraBridgeViewBase.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }

        }
    };
    //IMU
    private SensorManager mSensorManager;
    private float[] gyro = new float[3];
    //陀螺仪的旋转矩阵
    private float[] gyroMatrix = new float[9];
    //陀螺仪的方向角
    private float[] gyroOrientation = new float[3];
    private float[] magnet = new float[3];
    private float[] accel = new float[3];
    private float[] gravity = new float[3];
    private float[] accMagOrientation = new float[3];
    private float[] fusedOrientation = new float[3];
    //加速度计和磁力计的旋转矩阵
    private float[] rotationMatrix = new float[9];
    float[] initMatrix = new float[9];
    public static final float EPSILON = 0.000000001f;
    //纳秒转换为秒
    private static final float NS2S = 1.0f / 1000000000.0f;
    //毫秒转换为秒
    private static final float MS2S = 1 / 1000;
    private long timestamp;
    private long timestamp_vel;
    private boolean initState = true;
    public static final int TIME_CONSTANT = 30;
    //滤波系数
    public static final float FILTER_COEFFICIENT = 0.98f;
    private Timer fuseTimer = new Timer();
    public Handler mHandler;
    //    private TextView tv_imu,tv_cam;
//    DecimalFormat d = new DecimalFormat("#.##");//对数值的格式化方法
    float[] v_wi = new float[3];
    float[] p_wi = new float[3];
    float[] q_wi = new float[4];
    private double b_w = 0.053;
    private double b_a = 0.245;
    private double n_w = 1.19e-4;
    private double n_a = 1.77e-3;
    private float[] final_accel = new float[3];
    private float dT;

    float[] q_ic = new float[4];
    float[] p_ic = new float[3];

    StringBuffer Cam_buffer = new StringBuffer();
    String CamData;
    StringBuffer Imu_buffer = new StringBuffer();
    String ImuData;

    double[] fused_pq = new double[7];
    double[] fused_p = new double[3];
    double[] fused_q = new double[4];
    StringBuffer Fused_buffer = new StringBuffer();
    String fusedData;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        cameraBridgeViewBase = (CameraBridgeViewBase) findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);
        this.setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
        gyroOrientation[0] = 0.0f;
        gyroOrientation[1] = 0.0f;
        gyroOrientation[2] = 0.0f;

        gyroMatrix[0] = 1.0f; gyroMatrix[1] = 0.0f; gyroMatrix[2] = 0.0f;
        gyroMatrix[3] = 0.0f; gyroMatrix[4] = 1.0f; gyroMatrix[5] = 0.0f;
        gyroMatrix[6] = 0.0f; gyroMatrix[7] = 0.0f; gyroMatrix[8] = 1.0f;

        mSensorManager = (SensorManager) this.getSystemService(SENSOR_SERVICE);
        //注册传感器监听事件
        initListeners();
        //延时一秒计算合成的方向角
        fuseTimer.scheduleAtFixedRate(new calculateFusedOrientationTask(),1000, TIME_CONSTANT);
        mHandler = new Handler();

    }
    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found.Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_1_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package.Using it !");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        initListeners();
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
        mSensorManager.unregisterListener(this);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase != null) {
            cameraBridgeViewBase.disableView();
        }
        mSensorManager.unregisterListener(this);
//        saveCam(CamData);
//        saveIMU(ImuData);
        try {
            saveIMU(Imu_buffer.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    //visial code
    @Override
    public void onCameraViewStarted(int width, int height) {
        mRgb1 = new Mat();
        first_frame = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        mRgb1.release();
        first_frame.release();
    }

    double[] Rt_value = {0,0,0,0,0,0,0,0,0,0,0,0};
    double pre_time;
    double cur_time;
    @RequiresApi(api = Build.VERSION_CODES.CUPCAKE)
    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
//        mRgb1 = inputFrame.rgba();
        mGray = inputFrame.gray();
        if (numFrame == 0 && !mGray.empty()){
            mGray.copyTo(first_frame);
            pre_time = System.currentTimeMillis() * MS2S;
        }else {
            pre_p_wc = p_wc;
            pre_q_wc = q_wc;
            cur_time = System.currentTimeMillis() * MS2S;
            NativeFun.imuCoreFromJNI(p_wi,v_wi,q_wi,gyro,accel,dT);
            Rt_value = NativeFun.camMatrixFromJNI(mGray.getNativeObjAddr(),first_frame.getNativeObjAddr());
            if (Rt_value == null){
                return mGray;
            }else {
                p_wc[0] = (float) Rt_value[9]; p_wc[1] = (float) Rt_value[10]; p_wc[2] = (float) Rt_value[11];
                R_wc[0] = (float) Rt_value[0]; R_wc[1] = (float) Rt_value[1]; R_wc[2] = (float) Rt_value[2];
                R_wc[3] = (float) Rt_value[3]; R_wc[4] = (float) Rt_value[4]; R_wc[5] = (float) Rt_value[5];
                R_wc[6] = (float) Rt_value[6]; R_wc[7] = (float) Rt_value[7]; R_wc[8] = (float) Rt_value[8];
                SensorManager.getOrientation(R_wc,Vector_wc);
                float Z = (float) (Vector_wc[0] * 180/Math.PI);
                float X = (float) (Vector_wc[1] * 180/Math.PI);
                float Y = (float) (Vector_wc[2] * 180/Math.PI);
                q_wc = eulerAnglesToQuaternion(Z,X,Y);
//            double Cam_time = System.currentTimeMillis();
                CamData = "摄像头的位置与四元数"  + "\n" + "p_wc: " + p_wc[0] + "\t" + p_wc[1] + "\t" + p_wc[2] + "\n" + "q_wc: " + q_wc[0] + "\t" + q_wc[1] + "\t" + q_wc[2] + "\t" + q_wc[3] + "\n\n" ;
                Cam_buffer.append(CamData);
                try {
                    saveCam(Cam_buffer.toString());
                } catch (IOException e) {
                    e.printStackTrace();
                }
                fused_pq = NativeFun.FusedData();
                fused_p[0] = fused_pq[0]; fused_p[1] = fused_pq[1]; fused_p[2] = fused_pq[2];
                fused_q[0] = fused_pq[3]; fused_q[1] = fused_pq[4]; fused_q[2] = fused_pq[5]; fused_q[3] = fused_pq[6];
                fusedData = "融合后的位置与四元数" +  "\n" + "fused_p: " + fused_p[0] + "\t" + fused_p[1] + "\t" + fused_p[2] + "\n" + "fused_q: " + fused_q[0] + "\t" + fused_q[1] + "\t" + fused_q[2] + "\t" + fused_q[3] + "\n\n" ;
                Fused_buffer.append(fusedData);
                try {
                    saveFusedData(Fused_buffer.toString());
                } catch (IOException e) {
                    e.printStackTrace();
                }
             //Log.i("滤波后的坐标和四元数\n" ,"final_p: \n" +final_q[0] + "+  "+ final_q[1] +" * i +  "+final_q[2] +" * j +  "+ final_q[3] +" * k \n" +"p_wc:\n"+final_p[0]+"   "+final_p[1]+"   "+final_p[2]);

//            Log.i("相机的坐标和四元数\n" ,"q_wc: \n" +q_wc[0] + "+  "+ q_wc[1] +" * i +  "+q_wc[2] +" * j +  "+ q_wc[3] +" * k \n" +"p_wc:\n"+p_wc[0]+"   "+p_wc[1]+"   "+p_wc[2]);
//            Log.i("IMU的坐标和四元数\n","q_wi: \n" + q_wi[0] + "+  "+ q_wi[1] +" * i +  "+ q_wi[2] +" * j +  "+ q_wi[3] +" * k \n" +"p_wi:\n"+p_wi[0]+"   "+p_wi[1]+"   "+p_wi[2]);

//            //求取p_ic = p_wi - p_wc    q_ic = (q_wi)* * q_wc
//            p_ic[0] = (p_wi[0] - p_wc[0]);  p_ic[1] =  (p_wi[1] - p_wc[1]); p_ic[2] =  (p_wi[2] - p_wc[2]);
//            q_ic[0] = q_wi[0] * q_wc[0] - q_wi[1] * q_wc[1] - q_wi[2] * q_wc[2] - q_wi[3] * q_wc[3];
//            q_ic[1] = q_wi[0] * q_wc[1] + q_wi[1] * q_wc[0] + q_wi[2] * q_wc[3] - q_wi[3] * q_wc[2];
//            q_ic[2] = q_wi[0] * q_wc[2] + q_wi[2] * q_wc[0] + q_wi[3] * q_wc[1] - q_wi[1] * q_wc[3];
//            q_ic[3] = q_wi[0] * q_wc[3] + q_wi[3] * q_wc[1] + q_wi[2] * q_wc[3] - q_wi[2] * q_wc[1];
//           Log.i("相机和IMU的坐标和四元数","q_ic: \n" +q_ic[0] + "+  "+ q_ic[1] +" * i +  "+q_ic[2] +" * j +  "+ q_ic[3] +" * k \n" +"p_ic:\n"+p_ic[0]+"   "+p_ic[1]+"   "+p_ic[2]);
            }

        }
        numFrame++;
        NativeFun.camCoreFromJNI(p_wc,R_wc,pre_p_wc,pre_q_wc);
        return mGray;
    }
    private void initListeners() {

        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER), SensorManager.SENSOR_DELAY_UI);

        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE), SensorManager.SENSOR_DELAY_UI);

        mSensorManager.registerListener(this, mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD), SensorManager.SENSOR_DELAY_UI);
    }

    @Override
    public void onSensorChanged(SensorEvent event) {
        ImuData = "IMU的位置与四元数" +  "\n" + "p_wi: " + p_wi[0] + "\t" + p_wi[1] + "\t" + p_wi[2] + "\n" + "q_wi: " + q_wi[0] + "\t" + q_wi[1] + "\t" + q_wi[2] + "\t" + q_wi[3] + "\n\n" ;
        Imu_buffer.append(ImuData);

        switch (event.sensor.getType()){
            case Sensor.TYPE_ACCELEROMETER:
                System.arraycopy(event.values,0,accel,0,3);
                calculateAccMagOrientation();
                getvelocityandposition(event);
                NativeFun.evaluateIMU(p_wi,v_wi,q_wi,gyro,accel,dT);
                break;
            case Sensor.TYPE_GYROSCOPE:
                gyroFunction(event);
                break;
            case Sensor.TYPE_MAGNETIC_FIELD:
                System.arraycopy(event.values,0,magnet,0,3);
                break;
            default:
                break;
        }
    }

    @Override
    public void onAccuracyChanged(Sensor sensor, int accuracy) {

    }

    private void calculateAccMagOrientation() {
        if (SensorManager.getRotationMatrix(rotationMatrix,null,accel,magnet)){
            SensorManager.getOrientation(rotationMatrix,accMagOrientation);
        }
    }

    private void gyroFunction(SensorEvent event){
        if (accMagOrientation == null){
            return;
        }
        if (initState){
            initMatrix = getRotationMatrixFromOrientation(accMagOrientation);
            float[] test = new float[3];
            SensorManager.getOrientation(initMatrix,test);
            gyroMatrix = matrixMultiplication(gyroMatrix,initMatrix);
            initState = false;
        }
        float[] deltaVector = new float[4];
        if (timestamp != 0){
            final float dT = (event.timestamp - timestamp) * NS2S;
            System.arraycopy(event.values,0,gyro,0,3);
            getRotationVectorFromGyro(gyro, deltaVector, dT / 2.0f);
        }
        timestamp = (int)event.timestamp;
        float[] deltaMatrix = new float[9];
        SensorManager.getRotationMatrixFromVector(deltaMatrix,deltaVector);
        gyroMatrix = matrixMultiplication(gyroMatrix,deltaMatrix);
        SensorManager.getOrientation(gyroMatrix,gyroOrientation);
    }

    private void getRotationVectorFromGyro(float[] gyroValues, float[] deltaRotationVector, float timeFactor) {
        float[] normValues = new float[3];
        float omegaMagnitude = (float) Math.sqrt((gyroValues[0] * gyroValues[0] + gyroValues[1] * gyroValues[1] + gyroValues[2] * gyroValues[2]));
        if (omegaMagnitude > EPSILON){
            normValues[0] = gyroValues[0] / omegaMagnitude;
            normValues[1] = gyroValues[1] / omegaMagnitude;
            normValues[2] = gyroValues[2] / omegaMagnitude;
        }
        float thetaOverTwo = omegaMagnitude * timeFactor;
        float sinThetaOverTwo = (float) Math.sin(thetaOverTwo);
        float cosThetaOverTwo = (float) Math.cos(thetaOverTwo);
        deltaRotationVector[0] = sinThetaOverTwo * normValues[0];
        deltaRotationVector[1] = sinThetaOverTwo * normValues[1];
        deltaRotationVector[2] = sinThetaOverTwo * normValues[2];
        deltaRotationVector[3] = cosThetaOverTwo;
    }

    private float[] getRotationMatrixFromOrientation(float[] o) {
        float[] xM = new float[9];
        float[] yM = new float[9];
        float[] zM = new float[9];

        float sinX = (float) Math.sin(o[1]);
        float cosX = (float) Math.cos(o[1]);
        float sinY = (float) Math.sin(o[2]);
        float cosY = (float) Math.cos(o[2]);
        float sinZ = (float) Math.sin(o[0]);
        float cosZ = (float) Math.cos(o[0]);

        xM[0] = 1.0f; xM[1] = 0.0f;  xM[2] = 0.0f;
        xM[3] = 0.0f; xM[4] = cosX;  xM[5] = sinX;
        xM[6] = 0.0f; xM[7] = -sinX; xM[8] = cosX;

        yM[0] = cosY;  yM[1] = 0.0f; yM[2] = sinY;
        yM[3] = 0.0f;  yM[4] = 1.0f; yM[5] = 0.0f;
        yM[6] = -sinY; yM[7] = 0.0f; yM[8] = cosY;

        zM[0] = cosZ;  zM[1] = sinZ; zM[2] = 0.0f;
        zM[3] = -sinZ; zM[4] = cosZ; zM[5] = 0.0f;
        zM[6] = 0.0f;  zM[7] = 0.0f; zM[8] = 1.0f;

        //旋转顺序 y,x,z
        float[] resultMatrix = matrixMultiplication(xM,yM);
        resultMatrix = matrixMultiplication(zM,resultMatrix);
        return resultMatrix;
    }

    private float[] matrixMultiplication(float[] A, float[] B) {
        float[] result = new float[9];
        result[0] = A[0] * B[0] + A[1] * B[3] + A[2] * B[6];
        result[1] = A[0] * B[1] + A[1] * B[4] + A[2] * B[7];
        result[2] = A[0] * B[2] + A[1] * B[5] + A[2] * B[8];

        result[3] = A[3] * B[0] + A[4] * B[3] + A[5] * B[6];
        result[4] = A[3] * B[1] + A[4] * B[4] + A[5] * B[7];
        result[5] = A[3] * B[2] + A[4] * B[5] + A[5] * B[8];

        result[6] = A[6] * B[0] + A[7] * B[3] + A[8] * B[6];
        result[7] = A[6] * B[1] + A[7] * B[4] + A[8] * B[7];
        result[8] = A[6] * B[2] + A[7] * B[5] + A[8] * B[8];
        return result;
    }

    class calculateFusedOrientationTask extends TimerTask {
        @Override
        public void run() {
            float oneMinusCoeff = (float) (1.0 - FILTER_COEFFICIENT);
            if (gyroOrientation[0] < -0.5 * Math.PI && accMagOrientation[0] > 0.0) {
                fusedOrientation[0] = (float) (FILTER_COEFFICIENT * (gyroOrientation[0] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[0]);
                fusedOrientation[0] -= (fusedOrientation[0] > Math.PI) ? 2.0 * Math.PI : 0;
            }
            else if (accMagOrientation[0] < -0.5 * Math.PI && gyroOrientation[0] > 0.0) {
                fusedOrientation[0] = (float) (FILTER_COEFFICIENT * gyroOrientation[0] + oneMinusCoeff * (accMagOrientation[0] + 2.0 * Math.PI));
                fusedOrientation[0] -= (fusedOrientation[0] > Math.PI)? 2.0 * Math.PI : 0;
            }
            else {
                fusedOrientation[0] = FILTER_COEFFICIENT * gyroOrientation[0] + oneMinusCoeff * accMagOrientation[0];
            }

            // pitch
            if (gyroOrientation[1] < -0.5 * Math.PI && accMagOrientation[1] > 0.0) {
                fusedOrientation[1] = (float) (FILTER_COEFFICIENT * (gyroOrientation[1] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[1]);
                fusedOrientation[1] -= (fusedOrientation[1] > Math.PI) ? 2.0 * Math.PI : 0;
            }
            else if (accMagOrientation[1] < -0.5 * Math.PI && gyroOrientation[1] > 0.0) {
                fusedOrientation[1] = (float) (FILTER_COEFFICIENT * gyroOrientation[1] + oneMinusCoeff * (accMagOrientation[1] + 2.0 * Math.PI));
                fusedOrientation[1] -= (fusedOrientation[1] > Math.PI)? 2.0 * Math.PI : 0;
            }
            else {
                fusedOrientation[1] = FILTER_COEFFICIENT * gyroOrientation[1] + oneMinusCoeff * accMagOrientation[1];
            }

            // roll
            if (gyroOrientation[2] < -0.5 * Math.PI && accMagOrientation[2] > 0.0) {
                fusedOrientation[2] = (float) (FILTER_COEFFICIENT * (gyroOrientation[2] + 2.0 * Math.PI) + oneMinusCoeff * accMagOrientation[2]);
                fusedOrientation[2] -= (fusedOrientation[2] > Math.PI) ? 2.0 * Math.PI : 0;
            }
            else if (accMagOrientation[2] < -0.5 * Math.PI && gyroOrientation[2] > 0.0) {
                fusedOrientation[2] = (float) (FILTER_COEFFICIENT * gyroOrientation[2] + oneMinusCoeff * (accMagOrientation[2] + 2.0 * Math.PI));
                fusedOrientation[2] -= (fusedOrientation[2] > Math.PI)? 2.0 * Math.PI : 0;
            }
            else {
                fusedOrientation[2] = FILTER_COEFFICIENT * gyroOrientation[2] + oneMinusCoeff * accMagOrientation[2];
            }

            // 补偿陀螺仪漂移，
            gyroMatrix = getRotationMatrixFromOrientation(fusedOrientation);
            System.arraycopy(fusedOrientation, 0, gyroOrientation, 0, 3);
            // update sensor output in GUI
            mHandler.post(updateOrentationDisplayTask);
        }
    }

    private Runnable updateOrentationDisplayTask = new Runnable() {
        @Override
        public void run() {
            float hdg_Z =   (float) (fusedOrientation[0] * 180/Math.PI);
            float pitch_X = (float) (fusedOrientation[1] * 180/Math.PI);
            float roll_Y =  (float) (fusedOrientation[2] * 180/Math.PI);
            q_wi = eulerAnglesToQuaternion(hdg_Z,pitch_X,roll_Y);
        }
    };

    private void getvelocityandposition(SensorEvent event){
        if (fusedOrientation == null){
            return;
        }
        float[] linear_acceleration = new float[3];
        if (timestamp_vel != 0){
            dT = (event.timestamp - timestamp_vel) * NS2S;
            final_accel[0] = (float) (accel[0] - b_a - n_a);
            final_accel[1] = (float) (accel[1] - b_a - n_a);
            final_accel[2] = (float) (accel[2] - b_a - n_a);
            linear_acceleration[0] = initMatrix[0] * final_accel[0] + initMatrix[3] * final_accel[1] + initMatrix[6] * final_accel[2];
            linear_acceleration[1] = initMatrix[1] * final_accel[0] + initMatrix[4] * final_accel[1] + initMatrix[7] * final_accel[2];
            linear_acceleration[2] = initMatrix[2] * final_accel[0] + initMatrix[5] * final_accel[1] + initMatrix[8] * final_accel[2];

//            linear_acceleration[0] = initMatrix[0] * final_accel[0] + initMatrix[1] * final_accel[1] + initMatrix[2] * final_accel[2];
//            linear_acceleration[1] = initMatrix[3] * final_accel[0] + initMatrix[4] * final_accel[1] + initMatrix[5] * final_accel[2];
//            linear_acceleration[2] = initMatrix[6] * final_accel[0] + initMatrix[7] * final_accel[1] + initMatrix[8] * final_accel[2];
            v_wi[0] = (float) (v_wi[0] + linear_acceleration[0]* dT);
            v_wi[1] = (float) (v_wi[1] + linear_acceleration[1]* dT);
            v_wi[2] = (float) (v_wi[2] + (linear_acceleration[2] - 9.8 ) * dT);
            p_wi[0] = p_wi[0] + v_wi[0] * dT;
            p_wi[1] = p_wi[1] + v_wi[1] * dT;
            p_wi[2] = p_wi[2] + v_wi[2] * dT;
        }
        timestamp_vel = event.timestamp;

    }
    public float[] eulerAnglesToQuaternion(float Z,float X,float Y){
        float[] q = new float[4];
        float sinX = (float)Math.sin(X * 0.5f);
        float cosX = (float)Math.cos(X * 0.5f);
        float sinY = (float)Math.sin(Y * 0.5f);
        float cosY = (float)Math.cos(Y * 0.5f);
        float sinZ = (float)Math.sin(Z * 0.5f);
        float cosZ = (float)Math.cos(Z * 0.5f);
        q[0] = cosZ * cosX * cosY + sinZ * sinX * sinY;
        q[1] = sinZ * cosX * cosY - cosZ * sinX * sinY;
        q[2] = cosZ * sinX * cosY + sinZ * cosX * sinY;
        q[3] = cosZ * cosX * sinY - sinZ * sinX * cosY;
        return q;
    }

    public void saveCam(String data) throws IOException {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)){
            File SDcardDir = Environment.getExternalStorageDirectory(); //获取SD卡目录
            File saveFile = new File(SDcardDir,"data_camera.txt");
            FileOutputStream fileOutputStream = new FileOutputStream(saveFile);
            fileOutputStream.write(data.getBytes());
            fileOutputStream.close();
//            Toast.makeText(MainActivity.this,"camera data Saved",Toast.LENGTH_LONG).show();
        }else {
            Toast.makeText(this, "SD卡不存在", Toast.LENGTH_SHORT).show();
        }

    }
    public void saveIMU(String data) throws IOException {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)){
            File SDcardDir = Environment.getExternalStorageDirectory(); //获取SD卡目录
            File saveFile = new File(SDcardDir,"data_Imu.txt");
            FileOutputStream fileOutputStream = new FileOutputStream(saveFile);
            fileOutputStream.write(data.getBytes());
            fileOutputStream.close();
            Toast.makeText(MainActivity.this,"imu data Saved",Toast.LENGTH_LONG).show();
        }else {
            Toast.makeText(this, "SD卡不存在", Toast.LENGTH_SHORT).show();
        }
    }
    public void saveFusedData(String data) throws IOException {
        if (Environment.getExternalStorageState().equals(Environment.MEDIA_MOUNTED)){
            File SDcardDir = Environment.getExternalStorageDirectory(); //获取SD卡目录
            File saveFile = new File(SDcardDir,"data_fused.txt");
            FileOutputStream fileOutputStream = new FileOutputStream(saveFile);
            fileOutputStream.write(data.getBytes());
            fileOutputStream.close();
//            Toast.makeText(MainActivity.this,"fused data Saved",Toast.LENGTH_LONG).show();
        }else {
            Toast.makeText(this, "SD卡不存在", Toast.LENGTH_SHORT).show();
        }
    }


}