# -*- coding: utf-8 -*-
"""
bot_worker.py - محرك التحليل السحابي المخصص لـ GitHub Actions
"""
import os
import sys
import json
import time
import requests
from datetime import datetime

# رابط الـ Web App الخاص بـ Google Apps Script
# يفضل سحبه من GitHub Secrets، ووضع رابط افتراضي إن لم يوجد
GAS_API_URL = os.getenv("GAS_API_URL", "ضع_هنا_رابط_GOOGLE_APPS_SCRIPT_EXEC")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

def send_telegram_alert(pair, action, confidence, price):
    """إرسال إشعار فوري للتليجرام عند توفر إشارة قوية"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    emoji = "🟢" if action == "CALL" else "🔴"
    msg = (
        f"{emoji} <b>إشارة ذكاء اصطناعي جديدة!</b>\n\n"
        f"📊 <b>الزوج:</b> {pair}\n"
        f"🎯 <b>الاتجاه:</b> {action}\n"
        f"⚡ <b>نسبة الثقة:</b> {confidence}%\n"
        f"💵 <b>السعر:</b> {price}\n"
        f"⏰ <b>الوقت:</b> {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        print(f"Telegram error: {e}")

def push_to_google_sheet(pair, action, confidence, price):
    """إرسال النتيجة إلى Google Apps Script ليتم حفظها في Google Sheets"""
    payload = {
        "pair": pair,
        "action": action,
        "confidence": confidence,
        "price": price,
        "status": "PENDING",
        "source": "GitHub Actions v5"
    }
    
    print(f"جاري إرسال البيانات إلى Google Sheets: {payload}")
    try:
        response = requests.post(GAS_API_URL, json=payload, timeout=15)
        print("استجابة السيرفر:", response.text)
    except Exception as e:
        print(f"فشل الإرسال إلى Google Apps Script: {e}")

def run_analysis():
    print("بدء دورة تحليل السوق بواسطة نماذج الذكاء الاصطناعي...")
    
    # هنا يتم استدعاء خوارزميات التحليل الخاصة بك (مثل EMA / RSI / AI Classifier)
    # مثال توضيحي لمحاكاة توليد إشارة من الكود:
    pair = "EURUSD-OTC"
    action = "CALL"          # أو PUT حسب نتيجة المؤشرات
    confidence = 88.5        # النسبة المئوية
    price = 1.08450
    
    print(f"النتيجة: {pair} | {action} | دقة: {confidence}%")
    
    # حفظ الإشارة في Google Sheets
    push_to_google_sheet(pair, action, confidence, price)
    
    # إرسال للتليجرام
    send_telegram_alert(pair, action, confidence, price)

if __name__ == "__main__":
    if "ضع_هنا_رابط" in GAS_API_URL:
        print("خطأ: يرجى وضع رابط Google Apps Script أولاً!")
        sys.exit(1)
    run_analysis()