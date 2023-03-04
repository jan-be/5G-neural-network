import { useState } from 'react';
import './App.css';
import { dummyResponseValues, initialTestValues } from "./types";
import { getResponse } from "./apiConn";

function App() {
  const [textNumInputsObj, setTextNumInputsObj] = useState([...initialTestValues]);
  const [predNN, setPredNN] = useState(dummyResponseValues);
  const [predNs3, setPredNs3] = useState(dummyResponseValues);

  const send2ApiRequests = () => {
    getResponse(textNumInputsObj, false).then(e => setPredNN(e));
    getResponse(textNumInputsObj, true).then(e => setPredNs3(e));
  };

  return (
    <div>
      <h1>5G Network Simulator</h1>
      <div>
        <h3>Input</h3>

        <table>
          <tbody>
          {textNumInputsObj.map(param =>
            <tr key={param.name}>
              <td style={{textAlign: "right"}}>{param.name}{param.unit ? ` [${param.unit}]` : null}:</td>
              <td><input type="number" value={param.value ?? ""} onChange={e =>
                setTextNumInputsObj([...textNumInputsObj.filter(e => e.name !== param.name), {
                  name: param.name,
                  unit: param.unit,
                  value: Number(e.target.value),
                }])}/></td>
            </tr>)}
          </tbody>
        </table>
        <button onClick={send2ApiRequests}>Run Simulation</button>
      </div>

      {
        [{title: "Neural Network", arr: predNN}, {title: "5G-LENA", arr: predNs3}].map(e => <div>
          <h3>Output {e.title} Prediction</h3>

          <table>
            <tbody>
            {e.arr.map(param => <tr key={param.name}>
              <td style={{textAlign: "right"}}>{param.name}:</td>
              <td>{param.value?.toFixed(2)} {param.value ? param.unit : null}</td>
            </tr>)}
            </tbody>
          </table>
        </div>)
      }
    </div>
  );
}

export default App;
