using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FacialExpressions : MonoBehaviour {

    [Tooltip("Manually drag skin (H_DDS_HighRes) here.")]
    public GameObject skin;
    SkinnedMeshRenderer skinnedMeshRenderer;

    /*[Tooltip("Manually drag lower teeth (h_TeethDown) here.")]
    public GameObject teeth;
    SkinnedMeshRenderer skinnedMeshRendererTeeth;*/

    // to be controlled from outside this script:
    //int expression; // expressions are listet: 0 (neutral), 1 (happy), 2 (sad), 3 (angry), 4 (fearful), 5 (surprised)
    //float intensity = 1;
    //int blinkmax = 200; // maximum number of frames between blinks
    //int blinkmin = 50; // minimum number of frames between blinks
    //float lerpSpeed = 0.12f;

    int BlinkTicker = 0; // counts time since last blink
    int expressionLastFrame = 0; // is set to expression after each frame, so that change can be detected at start of each frame
    float intensityLastFrame = 1;
    float[] BlendshapesCurrent = new float[122]; //current facial expression, smoothly lerps into... 
    float[] BlendshapesGoal = new float[122];// goal facial expression, which is instantly set to one of the lists below. 

    // List of blendshapes for several facial expressions.
    // each value represents one facial blendshape (~ facial muscle) which can be controlled individually. 
    // zero-based array, so last value is Neutral[66]. Value 65 describes opening of mouth. Value 66 describes closing of eyes needed for blinking from this expression 
    // (this varies b/c a sad expression requires more closing of the eye to reach a closed position than a surprised expression etc.)

    // Blendshape matrix for eva
    /*float[] Neutral = new float[67] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0,  0, 90 //60 to 66. 
    };

    float[] Happy = new float[67] {
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0, 20, 20,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0, 30,  0, 80, 80, //20 to 29
        80, 80,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0, 50, 50,  0,  0,  0,  0, 80, 80, 80, //50 to 59
        80,  0,  0,  0,  0, 20, 80 //60 to 66
    };

    float[] Sad = new float[67] {
         0, 40,  0, 20,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0, 80, 80,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0, 40, //40 to 49
        40,  0,  0, 80, 80, 20, 20, 50, 50, 50, //50 to 59
        50,  0,  0,  0,  0,  0, 50 //60 to 66
    };

    float[] Angry = new float[67] {
         0,  0,  0,  0,  0,  0, 40,  0,  0,  0, //0 to 9
         0,  0,  0, 20, 20,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0, 20,  0,  0, 20,  0,  0, //20 to 29
         0,  0, 20, 20, 80, 80,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0, 60, 60,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,100,100,100,100, 40, 70 //60 to 66
    };

    float[] Fearful = new float[67] {
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0, 20,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0, 20,  0,  0, //20 to 29
         0,  0, 40, 40,  0,  0,  0,  0,  0,100, //30 to 39
         0,  0,  0,100,100,  0,  0,100,100,  0, //40 to 49
         0,  0,  0,  0,  0,100,100,100,100,100, //50 to 59
       100,  0,  0,  0,  0,  0,100 //60 to 66
    };

    float[] Surprise = new float[67] {
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,100,  0,  0,  0, //20 to 29
       100,100,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,100,100,  0, //40 to 49
         0, 20, 20,100,100, 20, 20,100,100,100, //50 to 59
       100,  0,  0,  0,  0,100,100 //60 to 66
    };

    float[] Disgust = new float[67] {
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
        0,  0,  0,  0,  0,  50,  50,  0,  0,  0, //10 to 19
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
        0,  0,  0,  0,  100,  100,  0,  0,  0,  0, //30 to 39
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
        0,  0,  0,  0,  0,  0, 90 //60 to 66. 
    };*/

    // Blendshape matrix for Amy, Amy does not have teeth skinned mesh renderer so the length for arrays is 102
    // These blendshape facial expressions are implemented according to EMFACS.

    /*float[] Neutral = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0 //100 to 101. 
    };

    float[] Happy = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  100,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 100, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0 //100 to 101.
    };

    float[] Sad = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  100,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 100, 0, 0, 0, 0, //90 to 99.
         0, 100 //100 to 101.
    };

    float[] Angry = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  100,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, -100, //70 to 79. 
         0, 0, 100, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 100, 0, 0, 0, 0, //90 to 99.
         0, 0 //100 to 101.
    };

    float[] Fearful = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  -100,  0,  0,  0, //30 to 39
         0,  50,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, -100, //70 to 79. 
         0, 0, 100, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 100, 0, 0, 100, 0, //90 to 99.
         0, 100 //100 to 101.
    };

    float[] Surprise = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  100,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, -50, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 100, 0, //90 to 99.
         0, 100 //100 to 101.
    };

    float[] Disgust = new float[102] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         100,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  100,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  100,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0 //100 to 101. 
    };*/

    float[] Neutral = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //100 to 109
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121. 
    };

    float[] Happy = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  100,  0,  0,  0,  0, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 100, 100, 0, 0, 0, 0, 0, 0, 0, //100 to 109
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    float[] Sad = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  100,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 100, //100 to 109
         0, 0, 0, 0, 100, 100, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    float[] Angry = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         100,  100,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 100, 100, 0, 0, 100, //80 to 89.
         100, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 100, //100 to 109
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    float[] Fearful = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  -100,  -100,  0,  0,  0,  0,  0,  100, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 100, 100, 0, 0, 100, //80 to 89.
         100, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 100, //100 to 109
         100, 100, 0, 0, 100, 100, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    float[] Surprise = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  0,  0,  0,  0,  0,  0,  0,  0,  100, //40 to 49
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 0, 0, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 100, 100, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //100 to 109
         100, 100, 0, 0, 100, 100, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    float[] Disgust = new float[122] { 
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //0 to 9
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //10 to 19
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //20 to 29
         0,  0,  0,  0,  0,  0,  0,  0,  0,  0, //30 to 39
         0,  100,  0,  0,  0,  0,  0,  0,  0,  0, //40 to 49
         0,  0,  100,  0,  0,  0,  0,  0,  0,  0, //50 to 59
         0,  0,  0,  0,  0, 0, 0, 0, 100, 100, //60 to 69. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //70 to 79. 
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //80 to 89.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //90 to 99.
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //100 to 109
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //110 to 119
         0, 0 //120 to 121.
    };

    // Use this for initialization
    void Start () {
        skinnedMeshRenderer = skin.GetComponent<SkinnedMeshRenderer>();
        //skinnedMeshRendererTeeth = teeth.GetComponent<SkinnedMeshRenderer>();
    }

    /// <summary>
    /// The actual expression function, which can be controlled from outside this script.
    /// The function is typically called every frame in the Update()-Loop. When calling is stopped, face freezes at current expression and blinking stops. 
    /// <param name="expression">expression index: 0 (neutral), 1 (happy), 2 (sad), 3 (angry), 4 (fearful), 5 (surprised). </param>
    /// <param name="intensity">value from 0 to 1, stating intensity of expression.</param>
    /// <param name="lerpSpeed">value from 0 to 1, stating speed of lerping from one expression to another. Close to 0: very small. Larger than 0.2: very fast.</param>
    /// <param name="blinkmin">minimum number of frames between eye blinks.</param>
    /// <param name="blinkmax">maximum number of frames between eye blinks.</param>
    /// </summary>
    public void Expression(int expression, float intensity, float lerpSpeed, int blinkmin, int blinkmax)
    {
        // STEP 1: set BlendshapesGoal when new Expression is selected, or when intensity is changed
        if ((expressionLastFrame != expression) || (intensityLastFrame != intensity))
        {
            if (intensity < 0) { intensity = 0; } // clamp intensity value
            if (intensity > 1) { intensity = 1; }

            expressionLastFrame = expression; // value is tracked to determine changes from frame to frame
            intensityLastFrame = intensity;


            //if (Expression == 1) { Ziel = Happy; } // setting entire array w/o looping does not work robustly
            if (expression == 4) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Neutral[i]; } } //up to value 67, the BlinkValue
            if (expression == 3) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Happy[i]; } }
            if (expression == 5) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Sad[i]; } }
            if (expression == 0) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Angry[i]; } }
            if (expression == 2) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Fearful[i]; } }
            if (expression == 6) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Surprise[i]; } }
            if (expression == 1) { for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = Disgust[i]; } }

            //for (int i = 0; i < BlendshapesGoal.GetLength(0); i++) { BlendshapesGoal[i] = BlendshapesGoal[i] * intensity; }
            for (int i = 0; i < 122; i++) { BlendshapesGoal[i] = BlendshapesGoal[i] * intensity; } // reduce if intesity < 1, but not for blinking value
        }


        // STEP 2: lerp current blendshapes towards goal
        for (int i = 0; i < 122; i++) { BlendshapesCurrent[i] = (BlendshapesCurrent[i] + (BlendshapesGoal[i] - BlendshapesCurrent[i]) * lerpSpeed); } // lerp expression in limited growth model (looks best here, imho). The larger lerpSpeed (currently .12), the faster the lerping. 
        for (int i = 0; i < 122; i++) { skinnedMeshRenderer.SetBlendShapeWeight(i, (int)BlendshapesCurrent[i]); } //... and actually run blendshapes (not number 65, that's for the mouth)
        //skinnedMeshRendererTeeth.SetBlendShapeWeight(29, (int)BlendshapesCurrent[65]); //run teeth blendshape. Number 29 of teeth blendshapes ist Teeth_mouthopen
        
        // STEP 3: Add blinking every once in a while
        /* BlinkTicker++;
        if ((BlinkTicker == 1) || (BlinkTicker == 3))
        {
            skinnedMeshRenderer.SetBlendShapeWeight(45, BlendshapesGoal[66] * 0.4f); 
            skinnedMeshRenderer.SetBlendShapeWeight(46, BlendshapesGoal[66] * 0.4f);
        }
        if ((BlinkTicker == 0) || (BlinkTicker == 4))
        {
            skinnedMeshRenderer.SetBlendShapeWeight(45, BlendshapesGoal[66] * 0.2f); 
            skinnedMeshRenderer.SetBlendShapeWeight(46, BlendshapesGoal[66] * 0.2f);
        }
        if (BlinkTicker == 2)
        {
            skinnedMeshRenderer.SetBlendShapeWeight(45, BlendshapesGoal[66]);
            skinnedMeshRenderer.SetBlendShapeWeight(46, BlendshapesGoal[66]);
        }
        if (BlinkTicker >= 5) { BlinkTicker = UnityEngine.Random.Range(-blinkmax, -blinkmin); } //Start counting for blinker from randomized point */
        
    }
}
