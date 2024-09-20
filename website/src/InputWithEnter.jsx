import React, { useState } from 'react';

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

export default InputWithEnter;