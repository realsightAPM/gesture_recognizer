Dataset of Leap Motion and Microsoft Kinect hand acquisitions used in:
G. Marin, F. Dominio, P. Zanuttigh, "Hand gesture recognition with Leap Motion and Kinect devices", IEEE International Conference on Image Processing (ICIP), Paris, France, 2014.

Department of Information Engineering - University of Padova (Italy)

June 5, 2014
____________________________

Files and directories structure:

- ./gestures.pdf:	Sketch of the gestures.

- ./kinect_calibration.mat: Kinect calibration parameters. Kinect calibration has been performed with the method of D. Herrera, J. Kannala, J. Heikkila, "Joint depth and color camera calibration with distortion correction", TPAMI, 2012.

- ./acquisitions/: Contains the acquisitions with this structure: Pp/Gg/ where 1<=p<=14 is the person and 1<=g<=10 is the gesture.

- ./acquisitions/Pp/Gg/:
    r_depth.png: Kinect depth map (640 x 480).
    r_depth.bin: Kinect depth map raw (640 x 480 short 16 bit. 0 means no valid value).
    r_rgb.png: Kinect color map (1280 x 960).
    r_leap_motion.cvs: Leap motion parameters.

    1<r=<=10 is the number of repetition.
____________________________

For additional information please write an e-mail to lttm@dei.unipd.it
