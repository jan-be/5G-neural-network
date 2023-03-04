import { Ns3Parameter } from "./types";

export async function getResponse(apiParameters: Ns3Parameter[], isNs3: boolean) {
  let resp = await fetch(`http://localhost:8000/predict/${isNs3 ? "ns3": "nn"}`, {
    method: "POST", body: JSON.stringify({params: apiParameters}),
    headers: {"Content-Type": "application/json"}
  });
  let json = await resp.json();
  return json.output;
}
