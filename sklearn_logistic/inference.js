const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const dataPath = 'data';
const resultsPath = 'results';
const testDataSize = 10;

async function main() {
    try {
        const session = await ort.InferenceSession.create('logistic.onnx');

        const testData = JSON.parse(fs.readFileSync(path.join(dataPath, '/X_test.json')).toString()).flat().flat();

        const feeds = {'input_0': [], 'input_1': [], 'input_2': [], 'input_3': []};

        testData.forEach((data, i) => {
            feeds[`input_${i % 4}`].push(data);
        });

        Object.keys(feeds).forEach(key => {
            feeds[key] = new ort.Tensor('float32', feeds[key], [testDataSize, 1]);
        });

        const results = await session.run(feeds);

        const result = results[session[['outputNames']]].data;
        const pyResults = JSON.parse(fs.readFileSync(path.join(resultsPath, '/py.json')).toString());

        const isSame = result.every((res, i) => Number(res) === pyResults[i]);

        return isSame;
    }
    catch (e) {
        console.log(`failed to inference ONNX model: ${e}.`);
        return false;
    }
}

main().then(isSame => {
    console.log(isSame ? 'Success: Result is same as python result' : 'Failure: Result is not the same as python result.');
});