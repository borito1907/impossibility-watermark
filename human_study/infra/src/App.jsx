import React, { useEffect, useMemo, useState } from 'react';

const InputWithEnter = ({ onEnterPress, user }) => {
  const [value, setValue] = useState(user);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // Prevents default form submission behavior
      onEnterPress(value);
    }
  };

  return (
    <input
      type="text"
      value={value}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={handleKeyDown}
      placeholder="Set your name."
      className="p-2 w-full sm:w-1/2 md:w-1/4 lg:w-1/6 border text-gray-500 caret-gray-500 border-gray-300 bg-gray-200 rounded"
    />
  );
};

function App() {
  const [user, setUser] = useState(() => localStorage.getItem('user') || '');
  const [data, setData] = useState({
    row: 0,
    total: 0,
    flip: null,
    prompt: '',
    promptID: 0,
    mutator: '',
    watermark: '',
    response_A: '',
    response_B: '',
    align_A: '',
    align_B: '',
    mutation_A: '',
    mutation_B: '',
  })
  const [answer1, setAnswer1] = useState('')
  const [answer2, setAnswer2] = useState('')
  const [choice, setChoice] = useState(null);
  const [submitted, setSubmitted] = useState(false);
  const [pending, setPending] = useState(false);
  const [mode, setMode] = useState(false);

  useEffect(() => {
    localStorage.setItem('user', user);
  }, [user]);

  const fetchQuestion = async (flip=data.flip) => {
    setSubmitted(false);
    const userid = encodeURIComponent(user)
    const response = await fetch(`/api/controlledtest/${userid}?flip=${flip}`);
    const res = await response.json();
    setData(res);
    setAnswers(res, undefined);
  }

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

  const handlePreference = (pref) => {
    if(choice === pref) {
      setChoice(null);
      return
    }
    setChoice(pref);
  };

  const handleSubmit = async () => {
    if(user === '') {
      alert('Please set your name in the upper right');
      return;
    }
    try {
      // Make a POST request with the selected answer
      setPending(true);
      const userid = encodeURIComponent(user);
      const post = await fetch(`/api/controlledtest/${userid}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ row: data.row, id: data.promptID, user, watermark: data.watermark, mutator: data.mutator, mutation_A: data.mutation_A, mutation_B: data.mutation_B, choice }),
      });
      setPending(false);

      if (!post.ok) {
        throw new Error('Network response was not ok');
      }
      setSubmitted(true);
    } catch (error) {
      alert(error)
    }
  };

  const handleNextQuestion = () => {
    let flip = data.flip;
    if(submitted) {
      flip = null
    } else {
      if(flip === true) flip = false
      else if(flip === false) flip = true
    }
    fetchQuestion(flip);
    setChoice(null);
  };

  const handleToggleMode = () => {
    const flip = !mode;
    setMode(flip);
    setAnswers(undefined, flip);
  };

  useMemo(fetchQuestion, [user]);

  return (
    <div className="min-h-screen w-screen bg-gray-100 p-4 flex flex-col">
      <div className="flex justify-end mb-4 mr-4 space-x-2">
        <span className='bg-gray-200 text-gray-500 rounded-full p-2'>{data.row}/{data.total}</span>
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
        <InputWithEnter onEnterPress={setUser} user={user}></InputWithEnter>
      </div>
      <div className="flex-1 flex flex-col items-center justify-center space-y-2 sm:space-y-4 w-full">
        <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
           <h2 className="text-sm sm:text-base md:text-lg text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: data.prompt}}></h2>
        </div>
        <div className="flex w-full max-w-6xl mx-auto space-x-2 sm:space-x-4">
          <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer1}}></div>
          <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer2}}></div>
        </div>
          {!submitted ?
        <div className="flex text-xs sm:text-sm md:text-base space-x-2 sm:space-x-4 max-w-6xl mx-auto justify-between mt-4">
          <button
            onClick={() => handlePreference('answer1')}
            className={`px-4 py-2 rounded ${choice === 'answer1' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
          >
            Answer 1
          </button>
          <button
            onClick={() => handlePreference('tie')}
            className={`px-4 py-2 rounded ${choice === 'tie' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
          >
            Tie
          </button>
          <button
            onClick={() => handlePreference('answer2')}
            className={`px-4 py-2 rounded ${choice === 'answer2' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
          >
            Answer 2
          </button>
          <button
            onClick={() => handlePreference('skipped')}
            className={`px-4 py-2 rounded ${choice === 'skipped' ? 'bg-blue-500 text-white' : 'bg-gray-400'}`}
          >
            Skip
          </button>
        </div>
        :
        <div className='space-y-0 text-center'>
          <span className="block text-gray-500">You selected {choice}. Answer 1 was mutation {data.mutation_A}. Answer 2 was mutation {data.mutation_B}</span>
          <span className="block text-gray-500">Prompt ID: {data.promptID}. Mutator used: {data.mutator}. Watermarker used: {data.watermark}</span>
        </div>
        }
        <div className="flex space-x-4 mt-4">
          <button
            onClick={choice !== null && !submitted ? handleSubmit : null}
            disabled={choice === null || submitted || pending}
            className={`px-4 py-2 ${choice !== null && !submitted && !pending? 'bg-blue-500' : 'cursor-not-allowed bg-gray-500'} text-white rounded`}
          >
            Submit
          </button>
          <button
            onClick={handleNextQuestion}
            className={`px-4 py-2 ${submitted ? 'bg-green-500' : 'bg-blue-500'} text-white rounded`}
          >
            {submitted ? 'Next Question' : 'Flip'}
          </button>
        </div>

      </div>
    </div>
  );
}

export default App;
