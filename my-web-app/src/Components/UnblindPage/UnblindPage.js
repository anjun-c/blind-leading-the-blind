import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './UnblindPage.css'; // Import the CSS file



const UnblindPage = () => {
  const [voiceRecording, setVoiceRecording] = useState(false);
  const [videoRecording, setVideoRecording] = useState(false);
  const [voiceSocket, setVoiceSocket] = useState(null);
  const [videoSocket, setVideoSocket] = useState(null);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [videoStatus, setVideoStatus] = useState('');
  const [videoStream, setVideoStream] = useState(null);
  const videoRef = useRef(null);
  const navigate = useNavigate();

  const handleVoiceRecord = async () => {
    if (!voiceRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const socket = new WebSocket('ws://127.0.0.1:8000/ws/stream_audio');
        setVoiceSocket(socket);

        socket.onopen = () => {
          console.log('Voice WebSocket connection opened');
          setVoiceStatus('Streaming audio...');
        };

        socket.onerror = (err) => {
          console.error('Voice WebSocket error: ', err);
          setVoiceStatus('Error in audio streaming.');
        };

        socket.onclose = () => {
          console.log('Voice WebSocket connection closed');
        };

        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
          }
        };
        recorder.start(1000);
        setVoiceRecording(true);
      } catch (err) {
        console.error('Error accessing microphone: ', err);
        setVoiceStatus('Error accessing microphone.');
      }
    } else {
      voiceSocket.close();
      setVoiceRecording(false);
      setVoiceStatus('Audio streaming stopped.');
    }
  };

  const handleVideoRecord = async () => {
    if (!videoRecording) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        setVideoStream(stream);
        videoRef.current.srcObject = stream;
        const socket = new WebSocket('ws://127.0.0.1:8000/ws/stream_video');
        setVideoSocket(socket);

        socket.onopen = () => {
          console.log('Video WebSocket connection opened');
          setVideoStatus('Streaming video...');
        };

        socket.onerror = (err) => {
          console.error('Video WebSocket error: ', err);
          setVideoStatus('Error in video streaming.');
        };

        socket.onclose = () => {
          console.log('Video WebSocket connection closed');
        };

        const recorder = new MediaRecorder(stream);
        recorder.ondataavailable = (e) => {
          if (e.data.size > 0 && socket.readyState === WebSocket.OPEN) {
            socket.send(e.data);
          }
        };
        recorder.start(1000);
        setVideoRecording(true);
      } catch (err) {
        console.error('Error accessing camera: ', err);
        setVideoStatus('Error accessing camera.');
      }
    } else {
      videoSocket.close();
      setVideoRecording(false);
      setVideoStatus('Video streaming stopped.');
      videoStream.getTracks().forEach(track => track.stop());
      setVideoStream(null);
    }
  };

  const handleBack = () => {
    navigate('/'); // Navigate back to the home page
  };

  return (
    <div className="unblind-container">

      <section>
        <h2>Voice Recording</h2>
        <button id="voiceRecordBtn" onClick={handleVoiceRecord}>
          {voiceRecording ? 'Stop Voice Recording' : 'Start Voice Recording'}
        </button>
        <p id="voiceStatus">{voiceStatus}</p>
      </section>

      <section>
        <h2>Video Recording</h2>
        <video ref={videoRef} id="videoPreview" width="320" height="240" autoPlay muted></video>
        <br />
        <button id="videoRecordBtn" onClick={handleVideoRecord}>
          {videoRecording ? 'Stop Video Recording' : 'Start Video Recording'}
        </button>
        <p id="videoStatus">{videoStatus}</p>
      </section>
    </div>
  );
};

export default UnblindPage;
