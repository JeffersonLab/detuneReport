import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import email.mime.application
import smtplib

from results import ResultSet


class EmailSender:

    def __init__(self, subject, fromaddr, toaddrs, smtp_server='localhost'):
        self.subject = subject
        self.fromaddr = fromaddr
        self.toaddrs = toaddrs
        self.smtp_server = smtp_server
        self.message = None

    def init_message(self):
        self.message = """
<html>
<head>
<style>
p {
  margin: 1px;
}
p .more_space {
  margin-bottom: 10px;
}
h1 {
  padding-left: 10px;
  padding-bottom: 4px;
  background-color: #342E63;
  color: white;
}
h2 {
  padding-left: 10px;
  padding-bottom: 4px;
  background-color: #52489C;
  color: white;
}
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding-left: 15px;
  padding-right: 15px;
  text-align: left;
}
th {
  background-color: #EBEBEB;
}
</style>
</head>
<body>
"""

    def finalize_message(self):
        self.message += """</body></html>"""

    def generate_message(self, rs: ResultSet):
        # Setup the header, etc.
        self.init_message()

        self.message += "<h1>Result Summary</h1>\n"
        self.message += rs.to_html_table_string()

        # Include the plots in the message.
        self.message += "<h1>Plots</h1>"
        for name in sorted(rs.epics_names):
            self.message += f"<h2>{name}</h2>"
            cav_images = rs.img_buffers[name]
            for run_num, img_buf in enumerate(cav_images):
                if img_buf is None:
                    self.message += f"<br><p>{name} run {run_num + 1} experienced"
                    self.message += f" an error.</p>"
                    self.message += f"<p>{rs.errors[name][run_num]}</p>\n"
                else:
                    string = base64.b64encode(img_buf.read()).decode()
                    uri = 'data:image/png;base64,' + string
                    self.message += f'<img width="400" height="200" '
                    self.message += f'src="{uri}"/>\n'

                    # Put the cursor back at start so they can be attachments.
                    img_buf.seek(0)

        self.finalize_message()

    def send_html_email(self, rs: ResultSet):
        self.generate_message(rs=rs)

        msg = MIMEMultipart('mixed')
        msg['Subject'] = self.subject
        msg['From'] = self.fromaddr
        msg['To'] = ",".join(self.toaddrs)

        part1 = MIMEText(self.message, 'html')
        msg.attach(part1)
        if rs.img_buffers is not None:
            for epics_name in sorted(rs.img_buffers.keys()):
                if rs.img_buffers[epics_name] is not None:
                    for img_buf in rs.img_buffers[epics_name]:
                        if img_buf is not None:
                            att = email.mime.image.MIMEImage(img_buf.read(),
                                                             _subtype='png')
                            att.add_header('Content-Disposition', 'attachment',
                                           filename=f"{epics_name}.png")
                            msg.attach(att)

        with smtplib.SMTP(self.smtp_server) as server:
            server.sendmail(msg['From'], self.toaddrs, msg.as_string())
