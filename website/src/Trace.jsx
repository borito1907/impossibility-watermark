import React, { useEffect, useState } from 'react';
import Papa from 'papaparse'; // Ensure this library is installed for CSV parsing

function Trace() {
    const [localData, setLocalData] = useState({
        raw_A: 'The quick brown fox jumps over the lazy dog.',
        raw_B: 'The really, really quick fox jumps over the crazy dog.',
        prompt: 'Prompt',
        title: 'Title',
        info: 'Info',
        stats: 'Stats'
    });
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
    
    const [steps, setSteps] = useState([]);
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [maxStep, setMaxStep] = useState(0);
    const [stepInput, setStepInput] = useState(0);
    
    // Begin copy
    const [answer1, setAnswer1] = useState('')
    const [answer2, setAnswer2] = useState('')
    const [mode, setMode] = useState(false);
    
    const fetchQuestion = async (raw_A = localData.raw_A, raw_B = localData.raw_B) => {
        const response = await fetch(`/api/diff?raw_A=${encodeURIComponent(raw_A)}&raw_B=${encodeURIComponent(raw_B)}`);
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
    // end copy
    
    // Handle CSV file upload and parse
    const handleFileUpload = (event) => {
        const file = event.target.files[0];
        if (file) {
            Papa.parse(file, {
                header: true,
                complete: (result) => {
                    setSteps(result.data);
                    setMaxStep(result.data.length - 2); // Maximum step is the length of data
                    setCurrentStepIndex(0); // Set initial step
                },
            });
        }
    };
    
    // Handle step changes based on input box
    const handleStepInputChange = (e) => {
        let value = parseInt(e.target.value, 10);
        if (isNaN(value)) {
            value = 0;
        }
        if (value >= 0 && value <= maxStep) {
            setStepInput(value);
            setCurrentStepIndex(value);
        }
    };
    
    // Handle navigation with buttons
    const handleStepChange = (increment) => {
        let newIndex = currentStepIndex + increment;
        if (newIndex >= 0 && newIndex <= maxStep) {
            setCurrentStepIndex(newIndex);
            setStepInput(newIndex); // Sync input with step change
        }
    };
    
    // Effect to update the text display when current step changes
    useEffect(() => {
        if (steps.length > 0) {
            const currentStep = steps[currentStepIndex];
            setLocalData({
                raw_A: currentStep.current_text,
                raw_B: currentStep.mutated_text,
                prompt: currentStep.prompt,
                title: `Step ${currentStep.step_num} - Mutation ${currentStep.mutation_num} - ${currentStep.quality_preserved === "True" && currentStep.length_issue==="False" ? "Pass" : "Fail"}`,
                info: `Length issue ${currentStep.length_issue} - Oracle pass ${currentStep.quality_preserved} - Backtrack ${currentStep.backtrack}`,
                stats: `Mutation time: ${currentStep.mutator_time} - Oracle time: ${currentStep.oracle_time} - Total time: ${currentStep.total_time}`
            });
            fetchQuestion(currentStep.current_text, currentStep.mutated_text);
        }
    }, [currentStepIndex, steps]);
    
    return (
        <div className="min-h-screen w-screen bg-gray-100 p-4 flex flex-col">
            {/* Upload CSV Button */}
            <div className="flex items-center justify-end mb-4 mr-4 space-x-2">
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
                <input type="file" accept=".csv" onChange={handleFileUpload} className="text-gray-500" />
            </div>

            {/* Display Current Step Information */}
            <div className="flex-1 flex flex-col items-center justify-center space-y-2 sm:space-y-4 w-full">
            
            {/* Step Input and Max Step Display */}
            <div className="flex justify-center items-center space-x-4 mb-4">
                <input 
                type="number"
                value={stepInput}
                onChange={handleStepInputChange}
                min={0}
                max={maxStep}
                className="border p-2 rounded bg-gray-200 text-gray-500"
                />
                <span className='text-gray-500 font-semibold'> / Max: {maxStep}</span>
            </div>

            {/* Step Navigation Buttons */}
            <div className="flex justify-center space-x-4 mb-4">
                <button className={`px-4 py-2 rounded ${currentStepIndex > 0 ? 'bg-blue-500 text-white' : 'bg-gray-400'}`} onClick={() => handleStepChange(-1)} disabled={currentStepIndex <= 0}>Previous Step</button>
                <button className={`px-4 py-2 rounded ${currentStepIndex < maxStep ? 'bg-blue-500 text-white' : 'bg-gray-400'}`} onClick={() => handleStepChange(1)} disabled={currentStepIndex >= maxStep}>Next Step</button>
            </div>
            <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
                <h2 className="text-sm sm:text-base md:text-lg text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: localData.prompt}}></h2>
            </div>
            <div className="bg-white p-2 sm:p-4 shadow rounded w-full max-w-6xl mx-auto">
                <h2 className="text-xs sm:text-sm md:text-base text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: localData.title}}></h2>
                <h2 className="text-xs sm:text-sm md:text-base text-gray-500 font-semibold text-center"dangerouslySetInnerHTML={{__html: localData.info}}></h2>
                <h2 className="text-xs sm:text-sm md:text-base text-gray-500 text-center"dangerouslySetInnerHTML={{__html: localData.stats}}></h2>
            </div>
            <div className="flex w-full max-w-6xl mx-auto space-x-2 sm:space-x-4">
                <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer1}}></div>
                <div className={`flex-1 text-xs sm:text-sm md:text-base font-mono text-gray-500 bg-white p-2 sm:p-4 shadow rounded ${mode?'break-all':''} whitespace-break-spaces`}dangerouslySetInnerHTML={{__html: answer2}}></div>
            </div>
        </div>
        
        </div>
    );
}

export default Trace;
