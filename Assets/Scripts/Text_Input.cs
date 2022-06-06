using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class Text_Input : MonoBehaviour {

    InputField input;
    InputField.SubmitEvent se;
    public Text output;
    string currentText;
    string newText;

	void Start () {

        // Reference text being typed into input box
        input = gameObject.GetComponent<InputField>();

        // Create UnityEvent that fires once Enter is pressed
        se = new InputField.SubmitEvent();

        // SubmitInput and TTS (within TTS Unity object) runs with InputField text as argument when se is fired
        se.AddListener(SubmitInput);
        se.AddListener(GameObject.Find("TTS").GetComponent<TTS_unity>().TTS);

        // Fire UnityEvent at end of editing (i.e. when Enter is pressed)
        input.onEndEdit = se;
		
	}

    private void SubmitInput(string arg)
    {

        // Keep what was in text box already
        if (output != null)
            currentText = output.text;
        else
            currentText = "";

        // Add recently input text to new line
        newText = currentText + "\n" + arg;

        // Keep what was in text box already
        if (output != null)
            output.text = newText;

        // clears input window
        input.text = "";

        // Re-activate input
        input.ActivateInputField();

    }
	
}
