"""
El. pašto siuntimo įrankis modelio treniravimo pranešimams
=========================================================

Naudojamas siunčiant pranešimus apie modelių treniravimo pabaigą.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate, make_msgid
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Įkrauname aplinkos kintamuosius iš .env failo
load_dotenv()


class EmailConfig:
    """El. pašto konfigūracija iš .env failo"""

    def __init__(self):
        self.smtp_server = os.environ.get('SMTP_SERVER')
        self.smtp_port = int(os.environ.get('SMTP_PORT', 587))
        self.smtp_username = os.environ.get('SMTP_USERNAME')
        self.smtp_password = os.environ.get('SMTP_PASSWORD')
        self.from_email = os.environ.get('SMTP_FROM_EMAIL')
        self.from_name = os.environ.get('SMTP_FROM_NAME', 'Duomenų Analizės Sistema')

    def is_enabled(self):
        """Patikrina, ar el. pašto siuntimas sukonfigūruotas"""
        return all([
            self.smtp_server,
            self.smtp_username,
            self.smtp_password,
            self.from_email
        ])


def format_duration(seconds):
    """
    Formatuoja trukmę į žmogui suprantamą formatą.

    Args:
        seconds: Trukmė sekundėmis

    Returns:
        str: Suformatuota trukmė (pvz., "1 val. 23 min. 45 sek.")
    """
    if seconds < 60:
        return f"{seconds:.1f} sek."
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes} min. {secs} sek."
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours} val. {minutes} min. {secs} sek."


def siusti_treniravimo_pranesima(
    modelio_pavadinimas: str,
    treniravimo_trukme: float,
    gavejo_email: str = "irmantas.pilypas@sa.stud.vu.lt",
    papildoma_info: dict = None
):
    """
    Siunčia el. laišką apie modelio treniravimo pabaigą.

    Args:
        modelio_pavadinimas: Modelio pavadinimas (pvz., "Random Forest", "XGBoost")
        treniravimo_trukme: Treniravimo trukmė sekundėmis
        gavejo_email: Gavėjo el. pašto adresas
        papildoma_info: Papildoma informacija (dict) apie treniravimą

    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Įkrauname konfigūraciją
        config = EmailConfig()

        if not config.is_enabled():
            return False, "El. pašto siuntimas nesukonfigūruotas (.env failas)"

        # Paruošiame el. laiško turinį
        pabaigos_laikas = datetime.now()
        pradzios_laikas = pabaigos_laikas - timedelta(seconds=treniravimo_trukme)

        trukme_formatuota = format_duration(treniravimo_trukme)

        # Papildoma informacija (paprasta lentelė)
        papildoma_html = ""
        if papildoma_info:
            papildoma_html = "<h3>Papildoma informacija</h3><table>"
            for raktas, reiksme in papildoma_info.items():
                papildoma_html += f"<tr><td><strong>{raktas}:</strong></td><td>{reiksme}</td></tr>"
            papildoma_html += "</table>"

        # HTML el. laiško turinys (paprastas dizainas)
        html_body = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    padding: 20px;
                    background-color: #f9f9f9;
                }}
                h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }}
                h3 {{
                    color: #34495e;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0 20px 0;
                    background-color: white;
                }}
                td {{
                    padding: 8px;
                    border-bottom: 1px solid #ddd;
                }}
                td:first-child {{
                    width: 180px;
                }}
                .success {{
                    background-color: #d4edda;
                    border: 1px solid #c3e6cb;
                    color: #155724;
                    padding: 12px;
                    margin: 15px 0;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 15px;
                    border-top: 1px solid #ccc;
                    color: #666;
                    font-size: 11px;
                }}
            </style>
        </head>
        <body>
            <h2>Modelio Treniravimas Baigtas</h2>

            <div class="success">
                <strong>{modelio_pavadinimas}</strong> modelio treniravimas sėkmingai baigtas!
            </div>

            <h3>Treniravimo informacija</h3>
            <table>
                <tr>
                    <td><strong>Modelio tipas:</strong></td>
                    <td>{modelio_pavadinimas}</td>
                </tr>
                <tr>
                    <td><strong>Treniravimo trukmė:</strong></td>
                    <td>{trukme_formatuota}</td>
                </tr>
                <tr>
                    <td><strong>Pradžios laikas:</strong></td>
                    <td>{pradzios_laikas.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
                <tr>
                    <td><strong>Pabaigos laikas:</strong></td>
                    <td>{pabaigos_laikas.strftime('%Y-%m-%d %H:%M:%S')}</td>
                </tr>
            </table>

            {papildoma_html}

            <div class="footer">
                <p>Šis el. laiškas buvo sugeneruotas automatiškai.</p>
                <p>Ekonominiu Rodikliu Imputacijos Tyrimas - Magistro darbas VU ŠA</p>
            </div>
        </body>
        </html>
        """

        # Sukuriame el. laišką
        msg = MIMEMultipart('alternative')
        msg['Subject'] = f"{modelio_pavadinimas} treniravimas baigtas - {trukme_formatuota}"
        msg['From'] = f"{config.from_name} <{config.from_email}>"
        msg['To'] = gavejo_email
        msg['Reply-To'] = config.from_email
        msg['Date'] = formatdate(localtime=True)
        msg['Message-ID'] = make_msgid(domain='reapi.lt')

        # Pridedame HTML turinį
        html_part = MIMEText(html_body, 'html', 'utf-8')
        msg.attach(html_part)

        # Siunčiame el. laišką
        print(f"Siunčiamas el. laiškas į {gavejo_email}...")
        with smtplib.SMTP(config.smtp_server, config.smtp_port) as server:
            server.starttls()
            server.login(config.smtp_username, config.smtp_password)
            server.send_message(msg)

        print(f"El. laiškas sėkmingai išsiųstas į {gavejo_email}")
        return True, f"El. laiškas sėkmingai išsiųstas į {gavejo_email}"

    except Exception as e:
        error_msg = f"Klaida siunčiant el. laišką: {str(e)}"
        print(f"KLAIDA: {error_msg}")
        import traceback
        traceback.print_exc()
        return False, error_msg


# Testavimo funkcija
if __name__ == "__main__":
    # Testuojame el. pašto siuntimą
    print("Testuojamas el. pašto siuntimas...")

    success, message = siusti_treniravimo_pranesima(
        modelio_pavadinimas="Random Forest (TEST)",
        treniravimo_trukme=125.5,  # 2 minutės 5.5 sekundės
        papildoma_info={
            "Rodiklių kiekis": 78,
            "Imputuotų reikšmių": "15,234",
            "CV R² vidurkis": "0.8542"
        }
    )

    if success:
        print(f"\nTESTAS PAVYKO: {message}")
    else:
        print(f"\nTESTAS NEPAVYKO: {message}")
