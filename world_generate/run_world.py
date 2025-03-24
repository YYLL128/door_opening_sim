#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import argparse
import subprocess


def find_xml_files(root_folder):
    """
    在给定的 root_folder 下，查找所有后缀为 .xml 的文件并返回它们的完整路径列表。
    """
    xml_paths = []
    for foldername, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".xml"):
                full_path = os.path.join(foldername, filename)
                xml_paths.append(full_path)
    return xml_paths


def main():
    parser = argparse.ArgumentParser(
        description="运行 Mujoco simulate，并指定一个xml文件；若不指定则随机选取。"
    )
    parser.add_argument(
        "--xml",
        type=str,
        default=None,
        help="要运行的xml文件路径（相对于world目录或绝对路径）。若不指定则随机运行。"
    )
    args = parser.parse_args()

    # 假设需要在 world 的三个子文件夹中查找 .xml 文件
    world_folder = os.path.join(os.path.dirname(__file__), "world")
    all_xml_files = find_xml_files(world_folder)

    if not all_xml_files:
        print("在 world 文件夹下未找到任何 .xml 文件，请检查路径。")
        return

    # 如果用户指定了 xml 文件，则尝试匹配或直接使用该路径
    chosen_xml = None
    if args.xml is not None:
        # 如果传入的是相对路径，那么把它与 world_folder 拼起来
        possible_path = os.path.join(world_folder, args.xml)

        if os.path.isfile(args.xml):
            # 如果本身就是绝对路径或脚本相对路径
            chosen_xml = os.path.abspath(args.xml)
        elif os.path.isfile(possible_path):
            # 如果是在 world 下的相对路径
            chosen_xml = os.path.abspath(possible_path)
        else:
            print(f"无法找到指定的文件: {args.xml}")
            return
    else:
        # 若没指定，则在所有文件中随机选取一个
        chosen_xml = random.choice(all_xml_files)

    # 打印一下要运行的文件，方便调试
    print(f"即将使用 simulate 运行的文件是: {chosen_xml}")

    # 通过 subprocess 调用 simulate 进程
    try:
        subprocess.run(["simulate", chosen_xml], check=True)
    except FileNotFoundError:
        print("未能找到 simulate 命令，请确认已将 MuJoCo 的 simulate 加入到系统 PATH 中，或使用绝对路径。")
    except subprocess.CalledProcessError as e:
        print(f"simulate 运行出错，返回码: {e.returncode}")


if __name__ == "__main__":
    main()
