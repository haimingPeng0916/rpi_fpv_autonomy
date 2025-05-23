#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <string.h>
#include <pthread.h>


#define MSP_STATUS 101
#define MSP_RAW_IMU 102
#define MSP_MOTOR 104
#define MSP_RC 105
#define MSP_RAW_GPS 106
#define MSP_COMP_GPS 107
#define MSP_ATT 108
#define MSP_ANALOG 110
#define MSP_PID 112


#define UART_DEVICE "/dev/ttyAMA0"


int init_uart(const char *dev, speed_t baud) {
    int fd = open(dev, O_RDWR | O_NOCTTY);
    if (fd < 0) { perror("open"); exit(1); }
    struct termios tio = {0};
    cfmakeraw(&tio);
    cfsetispeed(&tio, baud);
    cfsetospeed(&tio, baud);
    tio.c_cflag |= (CLOCAL | CREAD);
    tcsetattr(fd, TCSANOW, &tio);
    return fd;
}

// send request data to flight controller
void send_request_msp(int fd, uint8_t cmd) {
    uint8_t buf[6];
    buf[0] = "$"; buf[1]="M"; buf[2] = "<";
    buf[3]=0;
    buf[4]=cmd;
    buf[5] = buf[4] ^ buf[3];
    write(fd, buf, 6);
}

// send RC override command to flight controller
void send_control_msp(int fd, uint8_t cmd) {

}

// Read flight data after request it
int read_msp(int fd, uint8_t *cmd, uint8_t *payload, uint8_t *len) {
    uint8_t c;
    // sync on '$'
    while()
   
}

int main() {
    int fd = open_uart(, B115200);
    uint8_t cmd, len, payload[32];
    int cmds[4];


    while (1) {
        send_request_msp(fd, UART_DEVICE, MSP_ATT);
        if (read_msp(fd, &cmd, payload, &len)==0 && cmd==MSP_ATT && len==6) {
            int16_t r = payload[0] | (payload[1]<<8);
            int16_t p = payload[2] | (payload[3]<<8);
            int16_t y = payload[4] | (payload[5]<<8);
            printf("R: %.1f°, P: %.1f°, Y: %.1f°\n", r/100.0, p/100.0, y/100.0);
        }
        usleep(50000);
    }
    close(fd);
    return 0;
}
