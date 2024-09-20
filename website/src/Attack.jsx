import React, { useEffect, useState } from 'react';
import io from 'socket.io-client';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import InputWithEnter from './InputWithEnter';

const socket = io();

function Chat() {
    const [roomid, setRoomid] = useState('');
    const [mode, setMode] = useState(false);
    const [loading, setLoading] = useState(true);
    const [answer1, setAnswer1] = useState('')
    const [answer2, setAnswer2] = useState('')
    const [choice, setChoice] = useState(null);
    const [data, setData] = useState({
        prompt: '',
        taskid: -1,
        response_A: '',
        response_B: '',
        align_A: '',
        align_B: '',
    })
    
    useEffect(() => {
        socket.on('receive_task', (data) => {
            setData(data);
            setAnswers(data, undefined);
            setLoading(false);
        });
        
        socket.on('connect', () => {
            console.log('Connected to server');
        });
        
        socket.on('disconnected', (room) => {
            console.log('Disconnected from server');
            if(room === roomid) {
                setRoomid('');
                setLoading(true);
                alert(`You have been disconnected from room ${room}`);
            }
        });
        
        socket.on('joined', (room) => {
            toast.success(`Joined room ${room}`);
        });
        
        socket.on('error', (message) => {
            toast.error(message);
        });
        
        return () => {
            socket.off('receive_message');
            socket.off('connect');
        };
    }, []);
    
    const setAnswers = (res = data, align = mode) => {
        const responseA = res.response_A;
        const responseB = res.response_B;
        const alignA = res.align_A;
        const alignB = res.align_B;
        
        if (align) {
            setAnswer1(alignA);
            setAnswer2(alignB);
        } else {
            setAnswer1(responseA);
            setAnswer2(responseB);
        }
    }
    
    const handleToggleMode = () => {
        const flip = !mode;
        setMode(flip);
        setAnswers(undefined, flip);
    };
    
    const handleRoomid = (roomid) => {
        setRoomid(roomid);
        setLoading(true);
        socket.emit('join', {roomid});
    }
    
    const handleSubmit = () => {
        socket.emit('finish_task', {roomid, choice, taskid: data.taskid});
        setLoading(true);
        setChoice(null);
    };
    
    const handlePreference = (pref) => {
        if(choice === pref) {
            setChoice(null);
            return
        }
        setChoice(pref);
    };
    
    return (
        <div className="min-h-screen w-screen bg-gray-100 p-4 flex flex-col">
        <div className="flex justify-end mb-4 mr-4 space-x-2">
        <label className="flex items-center space-x-2">
        <span className='text-gray-500'>Align</span>
        <div className="relative">
        <input
        type="checkbox"
        checked={mode}
        onChange={handleToggleMode}
        className="sr-only"
        />
        <div className="block bg-gray-300 w-14 h-8 rounded-full"></div>
        <div
        className={`dot absolute left-1 top-1 bg-white w-6 h-6 rounded-full transition ${
            mode ? 'transform translate-x-6' : ''
        }`}
        ></div>
        </div>
        </label>
        <InputWithEnter onEnterPress={handleRoomid} user={roomid}></InputWithEnter>
        </div>
        <div className="flex-1 flex flex-col items-center justify-center space-y-2 sm:space-y-4 w-full">
        <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
        <h2 className="text-sm sm:text-base md:text-lg text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: data.prompt}}></h2>
        </div>
        <div className="flex w-full max-w-6xl mx-auto space-x-2 sm:space-x-4">
        <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer1}}></div>
        <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer2}}></div>
        </div>
        {!loading &&
            <>
            <div className="flex text-xs sm:text-sm md:text-base space-x-2 sm:space-x-4 max-w-6xl mx-auto justify-between mt-4">
            <button
            onClick={() => handlePreference('verybad')}
            className={`px-4 py-2 rounded ${choice === 'verybad' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
            >
            Very Bad
            </button>
            <button
            onClick={() => handlePreference('bad')}
            className={`px-4 py-2 rounded ${choice === 'bad' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
            >
            Bad
            </button>
            <button
            onClick={() => handlePreference('preserved')}
            className={`px-4 py-2 rounded ${choice === 'preserved' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
            >
            OK
            </button>
            </div>
            <div className="flex space-x-4 mt-4">
            <button
            onClick={choice !== null ? handleSubmit : () => {}}
            disabled={choice === null}
            className={`px-4 py-2 ${choice !== null ? 'bg-blue-500' : 'cursor-not-allowed bg-gray-500'} text-white rounded`}
            >
            Submit
            </button>
            </div>
            </>
        }
        <ToastContainer position="top-center" hideProgressBar/>
        </div>
        </div>
    );
}

export default Chat;
