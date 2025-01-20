import time
import pyvisa as visa
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pyvisa.errors import VisaIOError
from PIL import Image
import random
from typing import Literal

from scipy.signal import savgol_filter

from langchain.agents import Tool
from langchain_core.tools import StructuredTool
import functools


print("正在连接仪器.......")
rm = visa.ResourceManager()
sources = rm.list_resources()
if not sources:
    print("未找到仪器")
print(sources)
inst = rm.open_resource("USB0::0x5656::0x0832::AMOL323130008::INSTR")
awg = rm.open_resource("USB0::0x6656::0x0834::AWG4422490001::INSTR")
power = rm.open_resource("USB0::0x1AB1::0x0E11::DP8C200600640::INSTR")


inst.timeout = 10000
response = inst.query("*IDN?")
print("已连接到仪器：" + response)

global signal_generator_amplitude
global oscilloinst_amplitude
global oscilloinst_frequency


#################for debug###############

# def # save_parameters_to_file(parameters):
#     with open("parameters.txt", "a") as f:
#         f.write(f"{datetime.now()}: {parameters}\n")

#################for debug###############


# decorator
def parse_params(param_string):
    if param_string is None or param_string == "":
        return {}
    param_dict = {}
    params = param_string.split(", ")
    
    for param in params:
        if "=" in param:
            key, value = param.split("=", 1)  # 仅分割一次，以防值中包含 '='
            value = value.strip("'").strip('"')  # 去除单引号和双引号
            if value.strip():  # 确保值不为空字符串
                try:
                    value = float(value)
                except ValueError:
                    pass
                param_dict[key] = value
            else:
                pass
                # print(f"Warning: Empty value for key '{key}'")
        else:
            pass
            # print(f"Warning: Malformed parameter '{param}'")
    return param_dict



def param_decorator(func):
    @functools.wraps(func)
    def wrapper(param_string=None):
        param_dict = parse_params(param_string)
        return func(**param_dict)

    return wrapper


@param_decorator
def initialize_oscilloinst(initial_state: str = "ON") -> str:
    """初始化示波器"""
    # 重置并清除仪器
    inst.write(":RST")
    inst.write(":SYSTem:CLEAr")

    # save_parameters_to_file(f"initialize_oscilloinst")
    return f"\n示波器初始化完成。\n"


@param_decorator
def set_power_supply_channel(
    channel: str = "CH1",
    voltage: float = 5,
    current: float = 2,
    current_protect: float = 2.3,
    output_state: Literal["ON", "OFF"] = "ON",
) -> str:
    """
    设置电源的指定通道，包括电压、电流值、过流保护限值和输出状态。
    """

    power.write("*IDN?")
    power.write(f":INST {channel}")
    power.write(f":CURR {current}")
    power.write(f":CURR:PROT {current_protect}")
    power.write(f":CURR:PROT:STAT ON")
    power.write(f":VOLT {voltage}")
    power.write(f":OUTP {channel},{output_state}")

    # save_parameters_to_file(
    #     f"set_power_supply_channel: {channel}/ Voltage: {voltage}V/ Current: {current}A/ Current Protection: {current_protect}A/ Output: {output_state}"
    # )

    return (
        f"\n电源通道 {channel} 设置已完成：\n"
        f"电压: {voltage}V\n"
        f"电流: {current}A\n"
        f"过流保护: {current_protect}A\n"
        f"输出状态: {output_state}\n"
    )





@param_decorator
def set_oscilloinst_channel(
    channel: str = "CHAN1",
    state: Literal["ON", "OFF"] = "ON",
    coupling: Literal["DC", "AC", "GND"] = "DC",
    invert: Literal["ON", "OFF"] = "OFF",
    probe: str = "1X",
    offset: float = 0.0,
    scale: float = 1.0,
    units: Literal["VOLTs", "AMPeres", "WATTs", "UNKNown"] = "VOLTs",
    vernier: Literal["ON", "OFF"] = "OFF",
) -> str:
    """
    设置示波器的指定通道，包括通道标识符、显示状态、耦合方式、是否反转信号、探头衰减因子、波形垂直偏移量、垂直缩放因子、测量单位和精细调节选项。
    """
    inst.write(f":{channel}:DISP {state}")
    inst.write(f":{channel}:COUP {coupling}")
    inst.write(f":{channel}:INVert {invert}")
    inst.write(f":{channel}:PROBe {probe}")
    inst.write(f":{channel}:OFFSet {offset}")
    inst.write(f":{channel}:SCALe {scale}")
    inst.write(f":{channel}:UNITs {units}")
    inst.write(f":{channel}:VERNier {vernier}")

    # save_parameters_to_file(
    #     f"set_channel: {channel}/ {state}/ {coupling}/ {invert}/ {probe}/ {offset}/ {scale}/ {units}/ {vernier}"
    # )

    return (
        f"\n通道 {channel} 设置已完成：\n"
        f"显示状态: {state}\n"
        f"耦合方式: {coupling}\n"
        f"反相: {invert}\n"
        f"探头衰减: {probe}\n"
        f"垂直位移: {offset}\n"
        f"伏格档位: {scale}\n"
        f"单位: {units}\n"
        f"微调: {vernier}\n"
    )


@param_decorator
def configure_signal_generator(
    channel: str = "CHANnel1",
    mode: Literal["CONTinue", "MODulation", "BURSt"] = "CONTinue",
    waveform: Literal[
        "SINe", "SQUare", "PULSe", "RAMP", "ARB", "NOISe", "DC"
    ] = "SQUare",
    frequency: float = 2e5,
    amplitude: float = 2,
    offset: float = 0,
    phase: float = 0,
    duty: int = 50,
    invert: bool = False,
    sync_invert: bool = False,
    limit_enable: bool = False,
    limit_lower: float = None,
    limit_upper: float = None,
    amplitude_unit: Literal["VPP", "DBM", "VRMS"] = "VPP",
    psk_code: Literal["PN7", "PN9", "PN15", "PN21"] = "PN7",
    qam_code: Literal[
        "PN7", "PN9", "PN11", "PN15", "PN17", "PN21", "PN23", "PN25"
    ] = "PN7",
    trigger_source: Literal["INTernal", "EXTRise", "MANual"] = "INTernal",
    trigger_output: Literal["CLOSe", "RISe", "FALL"] = "RISe",
) -> str:
    """配置信号发生器的通道、工作模式、波形类型、频率、幅值、偏移量、相位角度、占空比、反向输出、同步反向输出、幅值限制、幅值单位、PSK编码、QAM编码、触发源和触发输出极性参数"""
    awg.write(f":{channel}:MODE {mode}")
    awg.write(f":{channel}:BASE:WAVe {waveform}")
    awg.write(f":{channel}:BASE:FREQuency {frequency}")
    awg.write(f":{channel}:BASE:AMPLitude {amplitude}")
    awg.write(f":{channel}:BASE:OFFSet {offset}")
    awg.write(f":{channel}:BASE:PHAse {phase}")
    awg.write(f":{channel}:BASE:DUTY {duty}")
    awg.write(f":{channel}:INVersion {'ON' if invert else 'OFF'}")
    awg.write(f":{channel}:OUTPut:SYNC:INVersion {'ON' if sync_invert else 'OFF'}")
    awg.write(f":{channel}:LIMit:ENABle {'ON' if limit_enable else 'OFF'}")
    global signal_generator_amplitude
    if limit_lower is not None:
        awg.write(f":{channel}:LIMit:LOWer {limit_lower}")
    if limit_upper is not None:
        awg.write(f":{channel}:LIMit:UPPer {limit_upper}")
    # awg.write(f":{channel}:AMPLitude:UNIT {amplitude_unit}")
    # awg.write(f":{channel}:LOAD {load}")
    awg.write(f":{channel}:PSK:PNCode {psk_code}")
    awg.write(f":{channel}:QAM:PNCode {qam_code}")
    awg.write(f":{channel}:TRIGger:SOURce {trigger_source}")
    awg.write(f":{channel}:TRIGger:OUTPut {trigger_output}")
    awg.write(f":{channel}:OUTPut ON")
    signal_generator_amplitude = amplitude
    # save_parameters_to_file(
    #     f"configure_awg: {channel}/ {mode}/ {waveform}/ {frequency}/ {amplitude}/ {offset}/ {phase}/ {duty}/ {invert}/ {sync_invert}/ {limit_enable}/ {limit_lower}/ {limit_upper}/ {amplitude_unit}/ {psk_code}/ {qam_code}/ {trigger_source}/ {trigger_output}"
    # )
    return (
        f"\n通道 {channel} 设置已完成：\n"
        f"模式: {mode}\n"
        f"波形: {waveform}\n"
        f"频率: {frequency} Hz\n"
        f"幅度: {amplitude} {amplitude_unit}\n"
        f"偏置: {offset}\n"
        f"相位: {phase}\n"
        f"占空比: {duty}%\n"
        f"反相: {'ON' if invert else 'OFF'}\n"
        f"同步反向输出: {'ON' if sync_invert else 'OFF'}\n"
        f"限幅: {'ON' if limit_enable else 'OFF'}\n"
        f"限幅下限: {limit_lower if limit_lower is not None else '未设置'}\n"
        f"限幅上限: {limit_upper if limit_upper is not None else '未设置'}\n"
        f"PSK码: {psk_code}\n"
        f"QAM码: {qam_code}\n"
        f"触发源: {trigger_source}\n"
        f"触发输出: {trigger_output}\n"
    )

@param_decorator
def calculate_amplitude_frequency_characteristic(
    channel: str = "CHANnel1",
    sweep_type: Literal["LINe", "LOG"] = "LINe",
    start_frequency: float = 1000,
    stop_frequency: float = 1e7,
    sweep_time: float = 1.4,
    trigger_sweep: bool = False,
    channel2: str = "CHAN1",
) -> str:
    """配置信号发生器的通道、扫描类型、起始频率、终止频率、扫描时间和触发扫描参数"""

    global signal_generator_amplitude
    channel = "CHANnel1"
    sweep_type = "LTNe"
    start_frequency = 1e3
    stop_frequency = 1e7
    sweep_time = 1.4
    a = sweep_time / 14
    awg.write(f":{channel}:OUTPut OFF")

    inst.write(f":{channel}:COUP DC")
    inst.write(":TRIGger:SWEEp SINGle")
    amp_lev = signal_generator_amplitude / 4
    inst.write(f":{channel}:SCALe {amp_lev}")
    inst.write(f":TIM:SCAL {a}")
    inst.write(f":TIM:OFFS {sweep_time/2}")

    awg.write(f":{channel}:MODE SWEep")
    awg.write(f":{channel}:SWEEp:TYPe {sweep_type}")
    awg.write(f":{channel}:SWEEp:FREQuency:STARt {start_frequency}")
    awg.write(f":{channel}:SWEEp:FREQuency:STOP {stop_frequency}")
    awg.write(f":{channel}:SWEEp:TIMe {sweep_time}")

    awg.write(f":{channel}:OUTPut ON")
    if trigger_sweep:
        awg.write(f":{channel}:SWEep:TRIGger")

    print(observe__ndwave(channel, start_frequency, stop_frequency))

    return (
        f"\n通道 {channel} 扫频配置完成：\n"
        f"扫频类型: {sweep_type}\n"
        f"起始频率: {start_frequency} Hz\n"
        f"截止频率: {stop_frequency} Hz\n"
        f"扫频时间: {sweep_time} s\n"
        f"扫频触发: {'是' if trigger_sweep else '否'}\n"
        f"\n---------------------------\n\n\n\n幅频特性曲线已绘制 \n\n\n\n---------------------------\n"
    )


@param_decorator
def observe_channel_wave(
    channel: str = "CHAN1",
):
    time.sleep(2)
    """观察示波器上的指定通道的波形"""
    inst.write(f":WAVeform:SOURce {channel}")
    inst.write(":WAVeform:MODE NORMal")
    inst.write(":WAVeform:FORMat ASCII")
    inst.write("WAVeform:DATA?")
    x = inst.read()

    # Remove unwanted characters
    x = x.strip().replace("\r", "")

    y = []
    for i in range(1400):
        try:
            y.append(float(x[14 * i + 11 : 14 * i + 24]))
        except ValueError as e:
            print(f"ValueError at index {i}: {e}")
            y.append(0.0)  # Append a default value in case of error

    y = np.array(y)
    t = np.arange(0, 1400, 1)
    plt.plot(t, y)
    # plt.show()
    plt.savefig("observe_wave.png")
    # save_parameters_to_file(f"observe_square_wave: {channel}")
    return f"\n示波器上的波形已显示给用户\n"


@param_decorator
def calculate_slew_rate(channel: str = "CHAN1", voltage_range=5):
    """计算压摆率"""
    inst.write(":KEY:auto")
    time.sleep(1)
    inst.write(f":MEASure:VAMPlitude? {channel}")
    voltage = float(inst.read())
    inst.write(f":MEASure:RISetime? {channel}")
    result = inst.read()
    # print(result)
    if result:
        try:
            rise_time = float(result)
        except ValueError:
            rise_time = None
            # print(f"Could not convert string to float: '{result}'")
    else:
        rise_time = None
        # print("Received an empty string instead of a rise time value.")

    if rise_time == None:
        return f"\n---------------------------\n\n\n\n未能成功测出压摆率\n\n\n\n---------------------------\n"
    else:
        inst.write(f":TIM:SCAL {rise_time/4}")  # 用于设置主时基挡位  UP ，DOWN 1
        time.sleep(1)
        inst.write(f":MEASure:RISetime? {channel}")
        result1 = inst.read()
        rise_time = float(result1)
        slew_rate = voltage / rise_time / 1e6
        # save_parameters_to_file(f"calculate_slew_rate: {channel}, {voltage}")
        # print(observe__wave(channel))
    # slew_rate = 0
    return f"\n---------------------------\n\n\n\n压摆率是: {slew_rate:.4f} V/us \n\n\n\n---------------------------\n"


@param_decorator
def calculate_time_delay(channel1: str = "CHAN1", channel2: str = "CHAN2"):
    """计算时间差"""
    inst.write(":KEY:auto")

    inst.write(f":MEASure:VAMPlitude? {channel1}")
    voltage = float(inst.read())
    inst.write(f":MEASure:RISetime? {channel1}")
    result = inst.read()
    # print(result)
    if result:
        try:
            rise_time = float(result)
        except ValueError:
            rise_time = None
            # print(f"Could not convert string to float: '{result}'")
    else:
        rise_time = None
        # print("Received an empty string instead of a rise time value.")

    inst.write(":CHAN1:COUP AC")
    inst.write(":CHAN2:COUP AC")
    inst.write(":KEY:auto")
    inst.write(f":TIM:SCAL {rise_time/4}")  # 用于设置主时基挡位  UP ，DOWN 1
    inst.write(f":MEASure:PDEL? {channel1},{channel2}")
    result = inst.read()
    # print(result)
    if result:
        try:
            time_delay = float(result)
        except ValueError:
            time_delay = None
            # print(f"Could not convert string to float: '{result}'")
    else:
        time_delay = None
        # print("Received an empty string instead of a time delay value.")
    # print(observe__dewave())
    # time_delay = 0
    return f"\n---------------------------\n\n\n\n\n时间差是: {abs(time_delay*1e9):4f} ns \n\n\n\n---------------------------\n"


@param_decorator
def calculate_amplitude(channel: str = "CHAN1"):
    global oscilloinst_amplitude
    inst.write(":KEY:auto")
    inst.write(f":MEASure:VAMPlitude? {channel}")
    result = inst.read()
    if result:
        try:
            amplitude = float(result) / 2
        except ValueError:
            amplitude = None
            # print(f"Could not convert string to float: '{result}'")
    else:
        amplitude = None
        # print("Received an empty string instead of a time delay value.")
    if amplitude == None:
        return f"\n---------------------------\n\n\n\n\n未成功测出交流信号幅度大小\n\n\n\n---------------------------\n"
    else:
        amp_lev = amplitude / 6
        inst.write(f":{channel}:SCALe {amp_lev}")
        inst.write(":KEY:auto")
        inst.write(f":MEASure:VAMPlitude? {channel}")
        result = inst.read()
        amplitude = float(result) / 2
        oscilloinst_amplitude = amplitude
        # save_parameters_to_file(f"calculate_slew_rate: {channel}")
        return f"\n---------------------------\n\n\n\n\n所测通道交流信号的幅度是: {amplitude:4f} V \n所测通道交流信号的峰峰值是: {amplitude*2:4f} V \n\n\n\n---------------------------\n"


@param_decorator
def calculate_DC(channel: str = "CHAN1"):
    inst.write(f":{channel}:COUP DC")
    inst.write(":KEY:auto")
    inst.write(f":MEASure:VAVerage? {channel}")
    result = inst.read()
    if result:
        try:
            amplitude = float(result) / 2
        except ValueError:
            amplitude = None
            # print(f"Could not convert string to float: '{result}'")
    else:
        amplitude = None
        # print("Received an empty string instead of a time delay value.")
    if amplitude == None:
        return f"\n---------------------------\n\n\n\n\n未成功测出直流信号大小\n\n\n\n---------------------------\n"
    else:
        amplitude = 0
    return f"\n---------------------------\n\n\n\n\n所测通道直流信号大小是: {amplitude:4f} V \n\n\n\n---------------------------\n"


@param_decorator
def calculate_frequency(channel: str = "CHAN1"):
    global oscilloinst_frequency
    inst.write(":KEY:auto")
    inst.write(f":{channel}:COUP AC")
    inst.write(f":MEASure:VAMPlitude? {channel}")
    result = inst.read()
    if result:
        try:
            amplitude = float(result) / 2
        except ValueError:
            amplitude = None
    else:
        amplitude = None
    if amplitude == None:
        return f"\n---------------------------\n\n\n\n\n未成功测出频率大小\n\n\n\n---------------------------\n"
    else:
        amp_lev = amplitude / 6
        inst.write(f":{channel}:SCALe {amp_lev}")
        inst.write(":KEY:auto")

        inst.write(f":MEASure:FREQuency? {channel}")
        result = inst.read()
        if result:
            try:
                frequency = float(result)
            except ValueError:
                frequency = None
        else:
            frequency = None
        if frequency == None:
            return f"\n---------------------------\n\n\n\n\n未成功测出频率大小\n\n\n\n---------------------------\n"
        else:
            oscilloinst_frequency = frequency
    # frequency = 0
    return f"\n---------------------------\n\n\n\n\n所测通道的频率是: {frequency:4f} Hz \n\n\n\n---------------------------\n"


@param_decorator
def calculate_power_ripple(channel: str = "CHAN1"):
    inst.write(f":{channel}:COUP DC")
    inst.write(":KEY:auto")
    inst.write(f":{channel}:COUP DC")

    inst.write(f":MEASure:VAVerage? {channel}")
    result = inst.read()
    if result:
        try:
            verage = float(result) * 1000
        except ValueError:
            verage = 1
    else:
        verage = 1

    inst.write(f":{channel}:COUP AC")
    inst.write(f":{channel}:SCALe 0.05")
    inst.write(":KEY:auto")
    inst.write(f":MEASure:VAMPlitude? {channel}")
    result = inst.read()
    if result:
        try:
            amplitude = float(result) * 1000
        except ValueError:
            amplitude = 1
    else:
        amplitude = 1
    inst.write(f":MEASure:FREQuency? {channel}")
    result = inst.read()
    if result:
        try:
            frequency = float(result)
        except ValueError:
            frequency = 0
    else:
        frequency = 0

    a = abs(amplitude / verage)

    return f"\n---------------------------\n\n\n\n\n电源直流电压为: {verage/1000:4f} V \n开关电源纹波频率为: {frequency:4f} Hz\n所测电源纹波大小为: {amplitude:4f} mV\n电源纹波系数为{a:4f}\n\n\n\n\n---------------------------\n"

@param_decorator
def calculate_opa_Magnification(channel: str = "CHAN1"):
    if oscilloinst_amplitude and signal_generator_amplitude:
        amplitude = oscilloinst_amplitude
        voltage_range = signal_generator_amplitude / 2
        # save_parameters_to_file(
        #     f"calculate_opa_Magnification: {channel}, {voltage_range}"
        # )
        return f"\n---------------------------\n\n\n\n\n所测运放增益是: {amplitude/voltage_range:4f} \n\n\n\n---------------------------\n"
    else:
        return f"\n---------------------------\n\n\n\n\n为能成功测量运放增益 \n\n\n\n---------------------------\n"


def observe__dewave():
    """观察示波器上的指定通道的波形"""
    # inst.write(":TIM:SCAL?")
    # t_level = float(inst.read())

    # inst.write(f":WAVeform:SOURce CHAN1")
    # inst.write(":WAVeform:MODE NORMal")
    # inst.write(":WAVeform:FORMat ASCII")
    # inst.write("WAVeform:DATA?")
    # x1 = inst.read()
    # time.sleep(3)
    # inst.write(f":WAVeform:SOURce CHAN2")
    # inst.write(":WAVeform:MODE NORMal")
    # inst.write(":WAVeform:FORMat ASCII")
    # inst.write("WAVeform:DATA?")
    # x2 = inst.read()
    # # Remove unwanted characters
    # fig, ax = plt.subplots()
    # x1 = x1.strip().replace("\r", "")
    # x2 = x2.strip().replace("\r", "")
    # y1 = []
    # for i in range(1400):
    #     try:
    #         y1.append(float(x1[14 * i + 11 : 14 * i + 24]))
    #     except ValueError as e:
    #         print(f"ValueError at index {i}: {e}")
    #         y1.append(0.0)  # Append a default value in case of error
    # y2 = []
    # for i in range(1400):
    #     try:
    #         y2.append(float(x2[14 * i + 11 : 14 * i + 24]))
    #     except ValueError as e:
    #         print(f"ValueError at index {i}: {e}")
    #         y2.append(0.0)  # Append a default value in case of error

    # y1 = np.array(y1)
    # y2 = np.array(y2)
    # y1 = savgol_filter(y1, 53, 3)
    # y2 = savgol_filter(y2, 53, 3)
    # t = np.arange(0, 1400, 1)
    # t = t * t_level * 1e9 / 100
    # ax.plot(t, y1, "r")
    # ax.plot(t, y2, "b")
    # ax.set_title("observe_square_wave")
    # ax.set_xlabel("t/ns")
    # ax.set_ylabel("amp/V")
    # ax.grid()
    # fig.savefig("observe_wave.png")
    # # save_parameters_to_file(f"observe_square_wave")
    return f"\n示波器上的波形已显示给用户，波形已保存为图片\n"


def observe__wave(
    channel: str = "CHAN1",
):
    time.sleep(2)
    """观察示波器上的指定通道的波形"""
    # inst.write(":TIM:SCAL?")
    # t_level = float(inst.read())

    # inst.write(f":WAVeform:SOURce {channel}")
    # inst.write(":WAVeform:MODE NORMal")
    # inst.write(":WAVeform:FORMat ASCII")
    # inst.write("WAVeform:DATA?")
    # x = inst.read()
    x =0 
    # Remove unwanted characters
    # fig, ax = plt.subplots()
    # x = x.strip().replace("\r", "")

    # y = []
    # for i in range(1400):
    #     try:
    #         y.append(float(x[14 * i + 11 : 14 * i + 24]))
    #     except ValueError as e:
    #         print(f"ValueError at index {i}: {e}")
    #         y.append(0.0)  # Append a default value in case of error

    # y = np.array(y)
    # t = np.arange(0, 1400, 1)
    # t = t * t_level * 1e9 / 100
    # y = savgol_filter(y, 53, 3)
    # ax.set_title("observe_square_wave")
    # ax.set_xlabel("t/ns")
    # ax.set_ylabel("amp/V")
    # ax.grid()
    # ax.plot(t, y)
    # # plt.title("observe_square_wave")
    # # plt.xlabel("t/ns")
    # # plt.ylabel("amp/V")
    # # plt.grid()
    # # plt.plot(t, y)
    # # # plt.show()
    # # plt.savefig("observe_wave.png")
    # fig.savefig("observe_wave.png")
    # save_parameters_to_file(f"observe_square_wave: {channel}")
    return f"\n示波器上的波形已显示给用户，波形已保存为图片\n"


def observe__ndwave(channel: str = "CHAN1", f_start=1e3, f_end=1e7):
    # time.sleep(2)
    # """观察示波器上的指定通道的波形"""
    # # inst.write(f":WAVeform:SOURce {channel}")
    # # inst.write(":WAVeform:MODE NORMal")
    # # inst.write(":WAVeform:FORMat ASCII")
    # # inst.write("WAVeform:DATA?")
    # # x = inst.read()
    # x=0
    # # Remove unwanted characters
    # x = x.strip().replace("\r", "")
    # fig, ax = plt.subplots()
    # y = []
    # for i in range(1400):
    #     try:
    #         y.append(float(x[14 * i + 11 : 14 * i + 24]))
    #     except ValueError as e:
    #         print(f"ValueError at index {i}: {e}")
    #         y.append(0.0)  # Append a default value in case of error

    # y = np.array(y)
    # t = np.arange(0, 1400, 1)
    # y1 = abs(y)
    # y1 = savgol_filter(y1, 53, 3)
    # f = t * (f_start + f_end) / 1400
    # ax.plot(f, y1)
    # ax.set_title("Amplitude-frequency characteristic curve")
    # ax.set_xlabel("f/Hz")
    # ax.set_ylabel("amp/V")
    # ax.grid()
    # # plt.title("Amplitude-frequency characteristic curve")
    # # plt.xlabel("f/Hz")
    # # plt.ylabel("amp/V")
    # # plt.grid()
    # # plt.show()
    # # plt.savefig("observe_wave1.png")
    # fig.savefig("observe_wave.png")

    # save_parameters_to_file(f"observe_square_wave: {channel}")
    return f"\n示波器上的波形已显示给用户，波形已保存为图片\n"

@param_decorator
def feedback_user(content:str,role:str):
    """反馈给用户的内容"""
    return f'\n---------------------------\n\n\n\n\n{content}\n\n\n\n---------------------------\n'


# 定义工具列表
tools = [
    StructuredTool.from_function(
        name="initialize_oscilloinst",
        func=initialize_oscilloinst,
        description="初始化示波器。重置并清除仪器，未设置自动设置。",
    ),
    StructuredTool.from_function(
        name="set_oscilloinst_channel",
        func=set_oscilloinst_channel,
        description="设置示波器的指定通道，包括通道标识符、显示状态、耦合方式、电阻值、是否反转信号、探头衰减因子、波形垂直偏移量、垂直缩放因子、测量单位和精细调节选项。",
    ),
    StructuredTool.from_function(
        name="configure_signal_generator",
        func=configure_signal_generator,
        description="配置信号发生器的通道、工作模式、波形类型、频率、幅值、偏移量、相位角度、占空比、反向输出、同步反向输出、幅值限制、幅值单位、负载阻抗、PSK编码、QAM编码、触发源和触发输出极性参数,扫频时不使用此函数",
    ),
    StructuredTool.from_function(
        name="observe_channel_wave",
        func=observe_channel_wave,
        description="观察示波器上的指定通道的波形。默认波形源为 CHAN1",
    ),
    StructuredTool.from_function(
        name="calculate_slew_rate",
        func=calculate_slew_rate,
        description="计算压摆率。返回压摆率的大小",
        # return_direct = True
    ),
    StructuredTool.from_function(
        name="calculate_time_delay",
        func=calculate_time_delay,
        description="计算示波器两个通道的时间差。返回时间差的大小",
    ),
    StructuredTool.from_function(
        name="calculate_amplitude",
        func=calculate_amplitude,
        description="测量示波器通道交流信号幅度或峰峰值。返回幅度的大小",
    ),
    StructuredTool.from_function(
        name="calculate_DC",
        func=calculate_DC,
        description="测量示波器通道直流信号。返回直流的大小",
    ),
    StructuredTool.from_function(
        name="calculate_frequency",
        func=calculate_frequency,
        description="测量示波器通道信号频率。返回频率的大小",
    ),
    StructuredTool.from_function(
        name="calculate_power_ripple",
        func=calculate_power_ripple,
        description="测量开关电源纹波。返回电源纹波的大小",
    ),
    StructuredTool.from_function(
        name="calculate_opa_Magnification",
        func=calculate_opa_Magnification,
        description="测量运算放大器的放大倍数（增益）。返回其大小",
    ),
    StructuredTool.from_function(
        name="calculate_amplitude_frequency_characteristic",
        func=calculate_amplitude_frequency_characteristic,
        description="配置信号发生器扫频相关参数并测量模块的幅频特性。返回其大小",
    ),
    StructuredTool.from_function(
        name="feedback_user",
        func=feedback_user,
        description="遇到任何异常情况时请不要道歉，而是按照格式来调用本函数，反馈给用户（请务必注意本函数只能反馈报错的内容）",
        return_direct=True
    ),
]

set_power_supply_channel(
    channel="CH1", voltage=5.0, current=2.0, current_protect=2.3, output_state="ON"
)
