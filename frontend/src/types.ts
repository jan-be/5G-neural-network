export interface Ns3Parameter {
  name: string,
  value?: number,
  unit?: string,
}

export const initialTestValues: Ns3Parameter[] = [
  {name: "numRowsGnb", value: 4},
  {name: "numColumnsGnb", value: 4},
  {name: "numRowsUe", value: 2},
  {name: "numColumnsUe", value: 2},
  {name: "gnbTxPower", unit: "dBm", value: 30},
  {name: "ueTxPower", unit: "dBm", value: 20},
  {name: "centralFrequency", unit: "Hz", value: 3600000000},
  {name: "bandwidth", unit: "Hz", value: 20000000},
  {name: "gnbUeDistance", unit: "m", value: 200},
];

export const dummyResponseValues: Ns3Parameter[] = [
  {name: "throughput", unit: "MBit/s"},
  {name: "delay", unit: "ms"},
];
