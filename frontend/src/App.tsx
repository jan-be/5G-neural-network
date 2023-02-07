import { useState } from 'react';
import './App.css';
import { dummyResponseValues, initialTestValues } from "./types";
import { getResponse } from "./apiConn";

function App() {
  const [textNumInputsObj, setTextNumInputsObj] = useState({...initialTestValues});
  const [predNN, setPredNN] = useState(dummyResponseValues);
  const [predNs3, setPredNs3] = useState(dummyResponseValues);

  const send2ApiRequests = () => {
    let numOb = Object.fromEntries(Object.entries(textNumInputsObj).map(([key,value]) => ([key, Number(value)])));

    getResponse(numOb, false).then(e => setPredNN(e));
    getResponse(numOb, true).then(e => setPredNs3(e));
  };

  return (
    <div>
      <h1>5G Network Simulator</h1>
      <div>
        <h3>Input</h3>

        <table>
          <tbody>
          {Object.entries(textNumInputsObj).map(([key, val]) =>
            <tr key={key}>
              <td>{key}:</td>
              <td><input type="number" value={val} onChange={e =>
                setTextNumInputsObj({...textNumInputsObj, [key]: e.target.value})}/></td>
            </tr>)}
          </tbody>
        </table>
        <button onClick={send2ApiRequests}>Get Response</button>
      </div>

      <div>
        <h3>Output neural network prediction</h3>

        <table>
          <tbody>
          {Object.entries(predNN).map(([key, val]) => <tr key={key}>
            <td>{key}:</td>
            <td>{val}</td>
          </tr>)}
          </tbody>
        </table>
      </div>

      <div>
        <h3>Output NS3 prediction</h3>

        <table>
          <tbody>
          {Object.entries(predNs3).map(([key, val]) => <tr key={key}>
            <td>{key}:</td>
            <td>{val}</td>
          </tr>)}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default App;
