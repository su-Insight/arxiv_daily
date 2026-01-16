#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试邮件发送功能的脚本

使用方法：
1. 修改脚本中的配置参数
2. 运行脚本：python test_email.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.construct_email import send_email
from loguru import logger


def test_send_email():
    """
    测试邮件发送功能
    """
    # 配置参数
    sender = "2784397857@qq.com"          # 发件人邮箱
    receiver = "2247493879@qq.com"        # 收件人邮箱
    password = ""           # 发件人邮箱密码或授权码
    smtp_server = "smtp.qq.com"           # SMTP服务器地址
    smtp_port = 465                             # SMTP服务器端口
    
    # 测试邮件内容
    html = """
    <html>
    <body>
        <h2>测试邮件</h2>
        <p>这是一封测试邮件，用于验证ArXiv Radar的邮件发送功能是否正常工作。</p>
        <p>如果您收到这封邮件，说明邮件发送功能已配置成功！</p>
        <br>
        <p>Best regards,<br>ArXiv Radar Team</p>
    </body>
    </html>
    """
    
    try:
        logger.info("开始发送测试邮件...")
        logger.info(f"发件人: {sender}")
        logger.info(f"收件人: {receiver}")
        logger.info(f"SMTP服务器: {smtp_server}:{smtp_port}")
        
        send_email(
            sender=sender,
            receiver=receiver,
            password=password,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            html=html
        )
        
        logger.success("测试邮件发送成功！请检查您的邮箱")
        return True
        
    except Exception as e:
        logger.error(f"测试邮件发送失败: {e}")
        logger.error("请检查以下配置是否正确:")
        logger.error("1. 发件人邮箱地址")
        logger.error("2. 发件人邮箱密码或授权码")
        logger.error("3. SMTP服务器地址和端口")
        logger.error("4. 网络连接是否正常")
        
        # 提供常见邮箱的SMTP配置参考
        logger.info("\n常见邮箱SMTP配置参考:")
        logger.info("Gmail:")
        logger.info("  SMTP服务器: smtp.gmail.com")
        logger.info("  端口: 587")
        logger.info("  注意: 需要启用'不太安全的应用访问'或使用应用专用密码")
        logger.info("\n163邮箱:")
        logger.info("  SMTP服务器: smtp.163.com")
        logger.info("  端口: 465")
        logger.info("  注意: 需要使用授权码登录")
        logger.info("\nQQ邮箱:")
        logger.info("  SMTP服务器: smtp.qq.com")
        logger.info("  端口: 465")
        logger.info("  注意: 需要使用授权码登录")
        
        return False


if __name__ == "__main__":
    logger.info("=== ArXiv Radar 邮件发送测试 ===")
    
    # 提示用户配置信息
    logger.warning("请确保已在test_email.py中正确配置以下信息:")
    logger.warning("- 发件人邮箱地址")
    logger.warning("- 发件人邮箱密码或授权码")
    logger.warning("- SMTP服务器地址和端口")
    
    input("\n配置完成后，按Enter键继续...")
    
    success = test_send_email()
    
    if success:
        logger.info("\n测试完成，邮件发送成功！")
    else:
        logger.info("\n测试完成，邮件发送失败。请检查错误信息并重新配置。")
        sys.exit(1)