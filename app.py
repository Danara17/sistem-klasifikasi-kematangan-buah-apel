from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

app = Flask(__name__)


def classify_apple(warna, diameter):
    # Fuzzy Membership Functions
    warna_apel = ctrl.Antecedent(np.arange(1, 4, 1), 'warna_apel')
    diameter_apel = ctrl.Antecedent(np.arange(4, 10, 1), 'diameter_apel')
    kematangan_apel = ctrl.Consequent(np.arange(0, 2, 1), 'kematangan_apel')

    warna_apel['hijau'] = fuzz.trimf(warna_apel.universe, [1, 1, 2])
    warna_apel['kuning'] = fuzz.trimf(warna_apel.universe, [1, 2, 3])
    warna_apel['merah'] = fuzz.trimf(warna_apel.universe, [2, 3, 3])

    diameter_apel['kecil'] = fuzz.trimf(diameter_apel.universe, [4, 4, 6])
    diameter_apel['sedang'] = fuzz.trimf(diameter_apel.universe, [4, 6, 7])
    diameter_apel['besar'] = fuzz.trimf(diameter_apel.universe, [6, 7, 9])

    kematangan_apel['belum_matang'] = fuzz.trimf(
        kematangan_apel.universe, [0, 0, 1])
    kematangan_apel['matang'] = fuzz.trimf(kematangan_apel.universe, [0, 1, 1])

    # Fuzzy Rule Base
    rule1 = ctrl.Rule(
        warna_apel['hijau'] & diameter_apel['kecil'], kematangan_apel['belum_matang'])
    rule2 = ctrl.Rule(
        warna_apel['hijau'] & diameter_apel['sedang'], kematangan_apel['belum_matang'])
    rule3 = ctrl.Rule(
        warna_apel['kuning'] & diameter_apel['sedang'], kematangan_apel['belum_matang'])
    rule4 = ctrl.Rule(
        warna_apel['kuning'] & diameter_apel['besar'], kematangan_apel['matang'])
    rule5 = ctrl.Rule(
        warna_apel['merah'] & diameter_apel['besar'], kematangan_apel['matang'])

    # Fuzzy Control System
    kematangan_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
    kematangan = ctrl.ControlSystemSimulation(kematangan_ctrl)

    kematangan.input['warna_apel'] = warna
    kematangan.input['diameter_apel'] = diameter

    # Fuzzy Inference
    kematangan.compute()

    return kematangan.output['kematangan_apel']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    warna = int(request.form['warna'])
    diameter = int(request.form['diameter'])
    klasifikasi = classify_apple(warna, diameter)

    if klasifikasi < 0.5:
        hasil_klasifikasi = '0 (belum matang)'
    else:
        hasil_klasifikasi = '1 (matang)'

    return render_template('result.html', klasifikasi=hasil_klasifikasi)


if __name__ == '__main__':
    app.run()
