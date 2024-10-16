const ort = require('onnxruntime-node');
const fs = require('fs');

const dataPath = 'data';
const resultsPath = 'results';

async function main() {
    try {
        // create a new session and load model.
        const session = await ort.InferenceSession.create('./unet.onnx');

        const testData = JSON.parse(fs.readFileSync(dataPath + '/X_test.json').toString()).flat().flat().flat();
        const imageArr = new Float32Array(testData);
        const tensor = new ort.Tensor('float32', imageArr, [5, 88, 120, 3]);

        // run inference with the specific inputs.
        const results = await session.run({'input_1': tensor});

        // get list of output names
        const result = results[session[['outputNames']]].data
        const pyResults = JSON.parse(fs.readFileSync(resultsPath + '/py.json').toString());

        // reshape from [5*88*120*3] to [5, 88, 120, 3]
        const resultReshaped = [];
        let index = 0;

        for (let i = 0; i < 5; i++) {
            let secondDimension = [];
            for (let j = 0; j < 88; j++) {
                let thirdDimension = [];
                for (let k = 0; k < 120; k++) {
                    thirdDimension.push(result.slice(index, index + 3));
                    index += 3;
                }
                secondDimension.push(thirdDimension);
            }
            resultReshaped.push(secondDimension);
        }

        let isSame = true;

        for (let i = 0; i < resultReshaped.length; i++) {
            for (let j = 0; j < resultReshaped[0].length; j++) {
                for (let k = 0; k < resultReshaped[0][0].length; k++) {
                    for (let l = 0; l < resultReshaped[0][0][0].length; l++) {
                        if (Math.abs(resultReshaped[i][j][k][l] - pyResults[i][j][k][l]) > 0.00001) {
                            isSame = false;
                            break;
                        }
                    }
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

