/*
Created by Youssef Elashry to allow two-way communication between Python3 and Unity to send and receive strings

Feel free to use this in your individual or commercial projects BUT make sure to reference me as: Two-way communication between Python 3 and Unity (C#) - Y. T. Elashry
It would be appreciated if you send me how you have used this in your projects (e.g. Machine Learning) at youssef.elashry@gmail.com

Use at your own risk
Use under the Apache License 2.0

Modified by: 
Youssef Elashry 12/2020 (replaced obsolete functions and improved further - works with Python as well)
Based on older work by Sandra Fang 2016 - Unity3D to MATLAB UDP communication - [url]http://msdn.microsoft.com/de-de/library/bb979228.aspx#ID0E3BAC[/url]
*/

using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using TMPro;

public class UdpSocket : MonoBehaviour
{
    [HideInInspector] public bool isTxStarted = false;

    [SerializeField] string IP = "127.0.0.1"; // local host
    [SerializeField] int rxPort = 8000; // port to receive data from Python on
    [SerializeField] int txPort = 8001; // port to send data to Python on

    public TMP_InputField mainInputField;
    public TMP_Text output; //Displaying AI sentence on the screen
    public GameObject loading;
    private List<string> userSentences; // A list of user typed sentences
    private int userNumSentence; // How many sentences user typed
    private int AINumSentence;
    private bool submitCheck; // Checking whether the sentence is successfully added to userSentences

    private int emotionLabel = -1; // emotionlabel={'anger':'0','disgust':'1','fear':'2','joy':'3','neutral':'4','sadness':'5','surprise':'6'}
    private string thisSentence;
    private List<string> receivedText; // Received text from python
    private int processedTexts;

    public GameObject Character;
    public GameObject TTSObject;
    public float userIntensity = 0.5f;
    public float userLerp = 0.012f;
    private FacialExpressions CharacterFacialExpressions = null;
    int expression = 4; // 4 is for neutral face

    private bool loadComplete = false; //Wait for Cerevoice to be loaded
    private bool first = true;

    Animator amyAnimator;


    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

    IEnumerator SendDataCoroutine()
    {
        while (true)
        {
            if(userNumSentence > 0 && submitCheck){
                SendData(userSentences[userNumSentence-1].ToString());
            }
            yield return new WaitForSeconds(1f);
        }
    }

    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            client.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    void Awake()
    {
        // User input part
        userSentences = new List<string>();
        receivedText = new List<string>();
        userNumSentence = 0;
        AINumSentence = 0;
        processedTexts = 0;
        submitCheck = false;

        mainInputField.onEndEdit.AddListener(SubmitName); // Adding user input function

        // UDP part
        
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Create local client
        client = new UdpClient(rxPort);

        // local endpoint define (where messages are received)
        // Create a new thread for reception of incoming messages
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();

        // Initialize (seen in comments window)
        print("UDP Comms Initialised");

        StartCoroutine(SendDataCoroutine()); // DELETE THIS: Added to show sending data from Unity to Python via UDP
        StartCoroutine(wait3secsForLoad());
    }

    void Start(){
        amyAnimator = GameObject.Find("Amy_Genesis2").GetComponent<Animator>();
        CharacterFacialExpressions = Character.GetComponent<FacialExpressions>();
        first = true;
    }

    void Update(){
        if(loadComplete){
            if(emotionLabel != -1){
                expression = emotionLabel;
            }
            CharacterFacialExpressions.Expression(expression, userIntensity, userLerp, 200, 400);
            

            if(thisSentence == "Hello! Stranger?"){
                if(first){
                    first = false;
                    TTSObject.GetComponent<TTS_unity>().TTS(thisSentence);
                    output.text = thisSentence;
                    processedTexts = 1;
                }
            }

            else if(AINumSentence > processedTexts){
                TTSObject.GetComponent<TTS_unity>().TTS(thisSentence);
                output.text = thisSentence;
                processedTexts = AINumSentence;

                amyAnimator.SetTrigger(expression.ToString());
            }

            /* if(first){
                TTSObject.GetComponent<TTS_unity>().TTS(thisSentence);
                output.text = thisSentence;
                first = false;
            }
            else if(receivedText[AINumSentence-1] != thisSentence){
                TTSObject.GetComponent<TTS_unity>().TTS(thisSentence);
                output.text = thisSentence;
            } */
        }
    }

    // Receive data, update packets received
    private void ReceiveData()
    {
        while (true)
        {
            try
            {
                IPEndPoint anyIP = new IPEndPoint(IPAddress.Any, 0);
                byte[] data = client.Receive(ref anyIP);
                string text = Encoding.UTF8.GetString(data);
                ProcessInput(text);
                AINumSentence++;
                //Debug.Log(AINumSentence);

                receivedText.Add(thisSentence);


                Debug.Log(">> emotionLabel: " + emotionLabel);
                Debug.Log(">> receivedText: " +receivedText[AINumSentence-1]);
                
            }
            catch (Exception err)
            {
                print(err.ToString());
            }
        }
    }

    private void ProcessInput(string input)
    {
        // PROCESS INPUT RECEIVED STRING HERE

        if (!isTxStarted) // First data arrived so tx started
        {
            isTxStarted = true;
        }

        emotionLabel = input[0] - '0';
        thisSentence = input.Substring(1);
    }

    //Prevent crashes - close clients and threads properly!
    void OnDisable()
    {
        if (receiveThread != null)
            receiveThread.Abort();

        client.Close();
    }

    private void SubmitName(string sentence)
    {
        submitCheck = false;
        userSentences.Add(sentence);
        userNumSentence++;
        submitCheck = true;
    }

    IEnumerator wait3secsForLoad(){
        loading.SetActive(true);
        yield return new WaitForSeconds(3f);
        loading.SetActive(false);
        loadComplete = true;
    }
}