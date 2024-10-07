import { Ollama } from "@langchain/ollama";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";

const model = new Ollama({
  model: "llama3.2", // Default value
  temperature: 0,
  maxRetries: 2,
  // other params...
});

const messages = [
  new SystemMessage("Translate the following from English into Italian"),
  new HumanMessage("hi!"),
];

console.log(await model.invoke(messages));

const parser = new StringOutputParser();
const result = await model.invoke(messages);
console.log(await parser.invoke(result));

const chain = model.pipe(parser);

console.log(await chain.invoke(messages));

const systemTemplate = "Translate the following into {language}:";
const promptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemTemplate],
  ["user", "{text}"],
]);

const promptValue = await promptTemplate.invoke({
  language: "italian",
  text: "hi",
});

//console.log(promptValue);

console.log(promptValue.toChatMessages());

const llmChain = promptTemplate.pipe(model).pipe(parser);
console.log(await llmChain.invoke({ language: "italian", text: "hi" }));