using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UMA;
using UMA.PoseTools;

public class TrollExtra : MonoBehaviour {

    public RuntimeAnimatorController animController;

    private UMAExpressionPlayer expressionPlayer;

	// Use this for initialization
	void Start () {

        // get reference to Troll avatar and UMAData
        UMADynamicAvatar umaDynamicAvatar = this.GetComponent<UMADynamicAvatar>();
        umaDynamicAvatar.Initialize();
        UMAData umaData = umaDynamicAvatar.umaData;

        // Fire callback once character created
        umaData.OnCharacterCreated += CharacterCreatedCallback;

        // Initialize Expression System and attach it to UMA data
        UMAExpressionSet expressionSet = umaData.umaRecipe.raceData.expressionSet;
        expressionPlayer = umaData.gameObject.AddComponent<UMAExpressionPlayer>();
        expressionPlayer.expressionSet = expressionSet;
        expressionPlayer.umaData = umaData;

        // Expose animation controls to external animator controller
        //umaDynamicAvatar.animationController = animController;

    }

    void CharacterCreatedCallback(UMAData umaData)
    {

        // Everything here can only occur once skeleton is initialized
        expressionPlayer.Initialize();

        // Enable realistic passive eye animation
        expressionPlayer.enableBlinking = true;
        expressionPlayer.minBlinkDelay = 1;
        expressionPlayer.maxBlinkDelay = 10;
        expressionPlayer.enableSaccades = true;

    }

    // Update is called once per frame
    void Update () {

        if (GameObject.Find("TTS").GetComponent<AudioSource>().isPlaying) // only run if speech is being output
        {
            if (expressionPlayer.gazeMode.ToString() != "Speaking")
            {
                expressionPlayer.gazeMode = ExpressionPlayer.GazeMode.Speaking;
            }
        }

        if (!GameObject.Find("TTS").GetComponent<AudioSource>().isPlaying)
        {
            if (expressionPlayer.gazeMode.ToString() != "Listening")
            {
                expressionPlayer.gazeMode = ExpressionPlayer.GazeMode.Listening;
            }
        }

    }
}