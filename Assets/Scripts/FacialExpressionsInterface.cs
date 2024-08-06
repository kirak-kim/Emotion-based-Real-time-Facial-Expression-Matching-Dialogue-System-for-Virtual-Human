using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FacialExpressionsInterface : MonoBehaviour {

    // Character
    [Tooltip("Manually drag your character here.")]
    public GameObject Character;
    public float userIntensity = 0.5f;
    public float userLerp = 0.012f;

    private FacialExpressions CharacterFacialExpressions = null;
    int expression = 0;

    // Use this for initialization
    void Start () {
        CharacterFacialExpressions = Character.GetComponent<FacialExpressions>();
    }

    void OnGUI()
    {
        GUILayout.BeginArea(new Rect(10, 10, 100, 200)); // You can change position of Interface here. This is designed so that all my interface scripts can run together. 
        if (GUILayout.Button("Neutral")) { expression = 4; }
        if (GUILayout.Button("Happy")) { expression = 3; }
        if (GUILayout.Button("Sad")) { expression = 5; }
        if (GUILayout.Button("Angry")) { expression = 0; }
        if (GUILayout.Button("Fearful")) { expression = 2; }
        if (GUILayout.Button("Surprised")) { expression = 6; }
        if (GUILayout.Button("Disgust")) { expression = 1; }
        GUILayout.EndArea();
    }

    // Update is called once per frame
    void Update () {
        //Expression(int expression, float intensity, float lerpSpeed, int blinkmin, int blinkmax)
        //CharacterFacialExpressions.Expression(expression, 1, 0.12f, 40, 200); Init
        //CharacterFacialExpressions.Expression(expression, 0.5f, 0.01f, 200, 400);
        CharacterFacialExpressions.Expression(expression, userIntensity, userLerp, 200, 400);
    }
}
