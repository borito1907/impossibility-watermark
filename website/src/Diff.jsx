import React, { useEffect, useMemo, useState } from 'react';

function Inspect() {
  const [localData, setLocalData] = useState({
    raw_A: 'The quick brown fox jumps over the lazy dog.',
    raw_B: 'The really, really quick fox jumps over the crazy dog.',
    title: 'Title'
  });
  const [data, setData] = useState({
    response_A: '',
    response_B: '',
    align_A: '',
    align_B: '',
  })
  const [answer1, setAnswer1] = useState('')
  const [answer2, setAnswer2] = useState('')
  const [mode, setMode] = useState(false);
  const [display, setDisplay] = useState(false);
  const [loading, setLoading] = useState(false);

  const fetchQuestion = async () => {
    const response = await fetch(`/api/diff?raw_A=${encodeURIComponent(localData.raw_A)}&raw_B=${encodeURIComponent(localData.raw_B)}`);
    const res = await response.json();
    setData(res);
    setAnswers(res, undefined);
    setLoading(false);
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

  const handleToggleMode = () => {
    const flip = !mode;
    setMode(flip);
    setAnswers(undefined, flip);
  };

  const handleToggleDisplay = () => {
    if(!display) {
      fetchQuestion();
      setLoading(true);
    } else {
      setLoading(false);
    }
    setDisplay(!display);
  }

  const handleInputChange = (field, value) => {
    setLocalData(prev => ({ ...prev, [field]: value }));
  };

  useMemo(fetchQuestion, []);

  return (
    <div className="min-h-screen w-screen bg-gray-100 p-4 flex flex-col">
      <div className="flex justify-end mb-4 mr-4 space-x-2">
        <button
          onClick={handleToggleDisplay}
          className="flex items-center justify-center p-2 bg-gray-300 hover:bg-gray-400 rounded-square"
        >
        {display ? 
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M7.127 22.562l-7.127 1.438 1.438-7.128 5.689 5.69zm1.414-1.414l11.228-11.225-5.69-5.692-11.227 11.227 5.689 5.69zm9.768-21.148l-2.816 2.817 5.691 5.691 2.816-2.819-5.691-5.689z"/></svg>
        :
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M15 12c0 1.654-1.346 3-3 3s-3-1.346-3-3 1.346-3 3-3 3 1.346 3 3zm9-.449s-4.252 8.449-11.985 8.449c-7.18 0-12.015-8.449-12.015-8.449s4.446-7.551 12.015-7.551c7.694 0 11.985 7.551 11.985 7.551zm-7 .449c0-2.757-2.243-5-5-5s-5 2.243-5 5 2.243 5 5 5 5-2.243 5-5z"/></svg>
        }
        </button>
        :
        
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
      </div>
      <div className="flex-1 flex flex-col items-center justify-center space-y-2 sm:space-y-4 w-full">
          {display ? <>
          <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
            <h2 className="text-sm sm:text-base md:text-lg text-gray-500 font-semibold text-center">{localData.title}</h2>
          </div>
          </>
          :
          <>
            <textarea
              className={`w-full max-w-6xl text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded`}
              value={localData.title}
              onChange={(e) => handleInputChange('title', e.target.value)}
              disabled={loading}
            />
          </>}
        <div className="flex w-full max-w-6xl mx-auto space-x-2 sm:space-x-4">
          {display && !loading ? <>
            <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer1}}></div>
            <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer2}}></div>
          </> :
          <>
            <textarea
              className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${loading?'opacity-60':''}`}
              value={localData.raw_A}
              onChange={(e) => handleInputChange('raw_A', e.target.value)}
              disabled={loading}
            />
            <textarea
              className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${loading?'opacity-60':''}`}
              value={localData.raw_B}
              onChange={(e) => handleInputChange('raw_B', e.target.value)}
              disabled={loading}
            />
          </>
          }
        </div>
      </div>
    </div>
  );
}

export default Inspect;
