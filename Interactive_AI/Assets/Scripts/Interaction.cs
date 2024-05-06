using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.Diagnostics; // Für die Process-Klasse
using System.IO; // Für die Verarbeitung der Ausgabe
using System.Threading.Tasks; // Für asynchrone Methoden
using System.Threading; // Für den UnitySynchronizationContext
#if UNITY_EDITOR
using UnityEditor; // Für den Zugriff auf Editor-spezifische Funktionen wie AssetDatabase
#endif

public class Interaction : MonoBehaviour
{
    private bool playerNearby = false;
    public AudioSource audioSource; // Füge eine öffentliche Variable für die AudioSource hinzu
    private SynchronizationContext unitySyncContext; // Um auf den Hauptthread zuzugreifen
    private AudioClip recordedClip; // Gespeicherter Clip für die Verarbeitung
    private bool isRecording = false; // Flag, um den Aufnahmezustand zu verfolgen

    void Awake()
    {
        unitySyncContext = SynchronizationContext.Current; // Initialisiere den UnitySynchronizationContext
    }

    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerNearby = true;
        }
    }

    void OnTriggerExit(Collider other)
    {
        if (other.CompareTag("Player"))
        {
            playerNearby = false;
        }
    }

    void Update()
    {
        if (playerNearby && Input.GetKeyDown(KeyCode.E))
        {
            if (!isRecording)
            {
                StartConversation();
            }
            else
            {
                StopRecording();
            }
        }
    }

    void StartConversation()
    {
        recordedClip = Microphone.Start(null, false, 10, 44100); // Startet die Aufnahme für max. 10 Sekunden
        isRecording = true;
    }

    void StopRecording()
    {
        Microphone.End(null);
        isRecording = false;
        SaveAudioClipToFile(recordedClip, "Assets/Audio/audioInput.wav");
        ExecuteActionAsync();
    }

    void SaveAudioClipToFile(AudioClip clip, string filePath)
    {
        if (clip == null) return;

        var samples = new float[clip.samples * clip.channels];
        clip.GetData(samples, 0);
        byte[] wavFile = ConvertAndWrite(samples, clip.channels, clip.frequency);
        File.WriteAllBytes(filePath, wavFile);

        #if UNITY_EDITOR
        AssetDatabase.Refresh(); // Refresh AssetDatabase to see the new file in the editor
        #endif
    }

    private byte[] ConvertAndWrite(float[] samples, int channels, int frequency)
    {
        MemoryStream stream = new MemoryStream();
        BinaryWriter writer = new BinaryWriter(stream);

        WriteWavHeader(writer, channels, frequency, samples.Length);
        foreach (var sample in samples)
        {
            var data = (short)(sample * short.MaxValue);
            writer.Write(data);
        }

        return stream.ToArray();
    }

    private void WriteWavHeader(BinaryWriter writer, int channels, int frequency, int samples)
    {
        writer.Write(new char[4] {'R', 'I', 'F', 'F'});
        writer.Write(36 + samples * 2);
        writer.Write(new char[4] {'W', 'A', 'V', 'E'});
        writer.Write(new char[4] {'f', 'm', 't', ' '});
        writer.Write(16);
        writer.Write((short)1);
        writer.Write((short)channels);
        writer.Write(frequency);
        writer.Write(frequency * channels * 2);
        writer.Write((short)(channels * 2));
        writer.Write((short)16);
        writer.Write(new char[4] {'d', 'a', 't', 'a'});
        writer.Write(samples * 2);
    }

    async void ExecuteActionAsync()
    {
        string output = await Task.Run(() => ExecuteAction());
        #if UNITY_EDITOR
        AssetDatabase.Refresh(); // Refresh the assets in the editor to properly load the audio
        #endif
        unitySyncContext.Post(_ => PlayAudio(), null);
    }

    string ExecuteAction()
    {
        string output = "";
        ProcessStartInfo startInfo = new ProcessStartInfo
        {
            FileName = "/Users/franz/miniforge3/envs/master/bin/python", //Pfad zur Python der Conda-Umgebung
            Arguments = "\"Assets/Scripts/text.py\"",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };

        using (Process process = new Process { StartInfo = startInfo })
        {
            process.Start();
            output = process.StandardOutput.ReadToEnd();
            string error = process.StandardError.ReadToEnd();
            process.WaitForExit();

            if (!string.IsNullOrEmpty(error))
            {
                UnityEngine.Debug.LogError("Python Error: " + error);
            }
        }

        if (!string.IsNullOrEmpty(output))
        {
            UnityEngine.Debug.Log("Python Output: " + output);
        }
        return output;
    }

    void PlayAudio()
    {
        if (audioSource != null)
        {
            audioSource.Play();
        }
    }
}
