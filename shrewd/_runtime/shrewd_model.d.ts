export interface ChoiceAnswer { choice: string; confidence: number; probabilities: Record<string, number> }
export interface ScoreAnswer { score: number; confidence: number; probabilities: Record<string, number> }
export interface NoulAnswer { noul: number; probabilities: Record<string, number> }
export type Answer = ChoiceAnswer | ScoreAnswer | NoulAnswer;
export type Answers = Record<string, Answer>;

export declare function readSafetensors(buffer: ArrayBuffer): Record<string, Float32Array | Float64Array>;

export declare class Model {
  readonly task: "classifier" | "panel";
  /** A classifier's labels, in the order predictProba uses. */
  readonly labels: string[] | null;
  readonly threshold: { value: number | null; questions?: string[] } | null;
  constructor(spec: object, weights: Record<string, Float32Array | Float64Array>);
  static load(folder: string): Promise<Model>;
  static fetch(baseUrl: string): Promise<Model>;
  predict(text: string): string;
  predict(texts: string[]): string[];
  predictProba(text: string): number[] | Record<string, number[]>;
  predictProba(texts: string[]): (number[] | Record<string, number[]>)[];
  decide(text: string): Answers;
  decide(texts: string[]): Answers[];
  confident(text: string): boolean;
  confident(texts: string[]): boolean[];
}
