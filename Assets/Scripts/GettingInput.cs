using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class GettingInput : MonoBehaviour
{
    public TMP_InputField mainInputField;
    public List<string> userSentences; // A list of user typed sentences
    public int numSentence; // How many sentences user typed

    void Start ()
    {
        numSentence = 0;

        mainInputField.onEndEdit.AddListener(SubmitName);
    }

    private void SubmitName(string sentence)
    {
        userSentences.Add(sentence);
        numSentence++;
    }
}
