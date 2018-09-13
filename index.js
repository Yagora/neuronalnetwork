'use strict'

let input = [];
let hidden = [];
let output = [];

let wh = [];
let wo = [];

let input_data = [1, 1, 0, 1];

function resetNetwork () {
    input = [0,0,0,0];
    hidden = [0,0,0,0];
    output = [0,0];

    wh = [[0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5]];

    wo = [[0.5, 0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5]];
}

function sigmoid (x) {
    return 1 / (1 + Math.pow(Math.E, (-1 * x)));
}

function propagate (d) {
    // input step
    for (var i = 0; i < input.length; i++) {
        input[i] = d[i];
    }
    // hidden step
    let xh = [0, 0, 0,0];
    for (var j = 0; j < hidden.length; j++) {
        for (var i = 0; i < input.length; i++) {
            xh[j] += wh[j][i];
        }
    }
    // activation step
    for (var j=0; j < hidden.length; j++) {
        hidden[j] = sigmoid(xh[j]);
    }

    // output step
    let xo = [0, 0];
    for (var k=0; k < output.length; k++) {
        for (var j = 0; j < hidden.length; j++) {
            xo[k] += wo[k][j] * hidden[j];
        }
    }
    // activation step
    for (var k = 0; k < output.length; k++) {
        output[k] = sigmoid(xo[k]);
    }
}

function main () {
    resetNetwork();
    propagate(input_data);
    console.log(output);
}

main();
