const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const dataPath = 'data';
const resultsPath = 'results';

async function main() {
    try {
        // create a new session and load the model
        const session = await ort.InferenceSession.create('./dummy_recurrent.onnx');
        console.log(session);

        // read test data from file json
        const testData = JSON.parse(fs.readFileSync(path.join(dataPath,'/X_test.json')).toString()).flat();
        const numeric = Float32Array.from(testData);
        const embedding_input = new ort.Tensor('float32', numeric, [10, 100]);

        const inputs = {
            'embedding_input': embedding_input
        }
        const results = await session.run(inputs);

        // compare
        const result = results[session[['outputNames']]].data;
        const pyResults = JSON.parse(fs.readFileSync(path.join(resultsPath,'py.json')).toString());

        //reshape to 10, 10
        const reshapedResult = [];
        for (let i = 0; i < result.length; i += 10) {
            reshapedResult.push(result.slice(i, i + 10));
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