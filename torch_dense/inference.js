const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const resultsPath = 'results';

async function main() {
    try {
        // create a new session and load the model.
        const session = await ort.InferenceSession.create('dense.onnx');
        console.log(session);

        const data1 = Float32Array.from([1.7640524, 0.4001572, 0.978738, 2.2408931]);
        const tensor1 = new ort.Tensor('float32', data1, [1, 4]);

        const feeds = {input: tensor1};
        const results = await session.run(feeds);

        const result = results[session.outputNames[0]].data;
        const pyResults = JSON.parse(fs.readFileSync(path.join(resultsPath,'py.json')).toString());

        let isSame = true;
        for (let i = 0; i < result.length; i++) {
            if (Math.abs(result[i] - pyResults[i]) > 0.0000001) {
                isSame = false;
                break;
            }
        }

        return isSame;

    } catch (e) {
        console.log(`failed to inference ONNX model: ${e}.`);
    }
}

main().then(isSame => {
    if (isSame) {
        console.log('Success: Result is same as python result (with precision 0.000001).');
    } else {
        console.log('Failure: Result is not the same as python result.');
    }
});