from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import email.mime.application
import smtplib

from results import ResultSet, ResultHTMLFormatter


class EmailSender:

    def __init__(self, subject, fromaddr, toaddrs, smtp_server='localhost'):
        self.subject = subject
        self.fromaddr = fromaddr
        self.toaddrs = toaddrs
        self.smtp_server = smtp_server
        self.message = None

    def send_html_email(self, rs: ResultSet):
        formatter = ResultHTMLFormatter()
        self.message = formatter.format(rs=rs)

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
                            # Make sure that we start at the front of the buffer.
                            img_buf.seek(0)
                            att = email.mime.image.MIMEImage(img_buf.read(),
                                                             _subtype='png')
                            att.add_header('Content-Disposition', 'attachment',
                                           filename=f"{epics_name}.png")
                            msg.attach(att)

        with smtplib.SMTP(self.smtp_server) as server:
            server.sendmail(msg['From'], self.toaddrs, msg.as_string())
