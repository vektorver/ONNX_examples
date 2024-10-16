const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const resultsPath = 'results';
const dataPath = 'data';

async function main() {
    try {
        // create a new session and load model.
        const session = await ort.InferenceSession.create('./multihead_attention.onnx');
        console.log(session);

        // read and prepare test data from file json
        const testData = JSON.parse(fs.readFileSync(path.join(dataPath, '/X_test.json')).toString());

        const inputs = {};
        const input = Float32Array.from(testData.flat().flat());
        inputs['input_1'] = new ort.Tensor('float32', input, [10, 11, 4]);

        // run the model
        const results = await session.run(inputs);
        const result = Array.from(results[session['outputNames']].data);
        const pyResults = JSON.parse(fs.readFileSync(path.join(resultsPath, '/py.json')).toString());

        //reshape
        const reshapedResult = [];
        for (let i = 0; i < result.length; i += 4) {
            reshapedResult.push(result.slice(i, i + 4));
        }

        let isSame = true;
        for (let i = 0; i < reshapedResult.length; i++) {
            for (let j = 0; j < reshapedResult[i].length; j++) {
                if (Math.abs(reshapedResult[i][j] - pyResults[i][j]) > 0.000001) {
                    isSame = false;
                    break;
                }
            }
        }

        return isSame;

    } catch (e) {
        console.log(`failed to inference ONNX model: ${e}.`);
        return false;
    }
}

main().then(isSame => {
    if (isSame) {
        console.log('Success: Result is same as python result (with precision 0.000001).');
    } else {
        console.log('Failure: Result is not the same as python result.');
    }
});