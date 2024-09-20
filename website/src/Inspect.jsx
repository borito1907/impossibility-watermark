import React, { useEffect, useMemo, useState } from 'react';
import { useParams } from 'react-router-dom';

function Inspect() {
  const { id } = useParams();
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
  const [mode, setMode] = useState(false);

  const fetchQuestion = async () => {
    const response = await fetch(`/api/getrow/${id}`);
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

  const handleToggleMode = () => {
    const flip = !mode;
    setMode(flip);
    setAnswers(undefined, flip);
  };

  useMemo(fetchQuestion, [id]);

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
      </div>
      <div className="flex-1 flex flex-col items-center justify-center space-y-2 sm:space-y-4 w-full">
        <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
           <h2 className="text-sm sm:text-base md:text-lg text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: data.prompt}}></h2>
        </div>
        <div className="flex w-full max-w-6xl mx-auto space-x-2 sm:space-x-4">
          <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer1}}></div>
          <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer2}}></div>
        </div>
        <div className='space-y-0 text-center'>
          <span className="block text-gray-500">Answer 1 was mutation {data.mutation_A}. Answer 2 was mutation {data.mutation_B}</span>
          <span className="block text-gray-500">Prompt ID: {data.promptID}. Mutator used: {data.mutator}. Watermarker used: {data.watermark}</span>
        </div>
      </div>
    </div>
  );
}

export default Inspect;
