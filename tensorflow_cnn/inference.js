const ort = require('onnxruntime-node');
const fs = require("fs");
const path = require('path');

const dataPath = 'data';
const resultsPath = 'results';

async function main() {
    try {
        // create a new session and load the specific model.
        const session = await ort.InferenceSession.create('./cnn.onnx');
        console.log(session);

        const testData = Float32Array.from(JSON.parse(fs.readFileSync(path.join(dataPath, 'x_test.json')).toString()).flat().flat().flat());
        const tensor1 = new ort.Tensor('float32', testData, [10, 32, 32, 3]);
        const feeds = {inp: tensor1};

        // feed inputs and run
        const results = await session.run(feeds);

        const result = results[session[['outputNames']]].data
        const pyResults = JSON.parse(fs.readFileSync(path.join(resultsPath, 'py.json')).toString());

        const resultReshaped = [];

        for (let i = 0; i < 10; i++) {
            resultReshaped.push([]);
            for (let j = 0; j < 10; j++) {
                resultReshaped[i].push(result[i * 10 + j]);
            }

        }

        let isSame = true;

        for (let i = 0; i < resultReshaped.length; i++) {
            for (let j = 0; j < resultReshaped[0].length; j++) {
                if (Math.abs(resultReshaped[i][j] - pyResults[i][j]) > 0.00001) {
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
        console.log('Success: Results are the same as python results (with precision 0.00001).');
    } else {
        console.log('Results are not the same as python results.');
    }
});