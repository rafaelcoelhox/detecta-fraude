// Escuta TCP em :9999 e, para cada conexão aceita, envia o file descriptor
// (via SCM_RIGHTS) para uma das APIs em /sockets/api{N}.sock — round-robin.
// Nunca lê bytes da conexão, nunca inspeciona payload, nunca responde HTTP.
//
// Compilação: gcc -O2 -Wall -Wextra -o fd-lb fd-lb.c
//
// Variáveis de ambiente:
//   LB_PORT      — porta TCP (default 9999)
//   LB_BACKLOG   — backlog do listen (default 4096)
//   API_SOCKETS  — sockets separados por vírgula (default /sockets/api1.sock,/sockets/api2.sock)
//   READY_DIR    — diretório onde aparecem arquivos <name>.ready (default /sockets)

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#define MAX_BACKENDS 32

static int connect_backend(const char *path) {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    struct sockaddr_un addr = {0};
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);
    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return fd;
}

static int wait_for_socket(const char *path) {
    // Aguarda o arquivo do socket existir (a API cria após o mmap warm-up).
    int tries = 0;
    while (tries++ < 600) {
        struct stat st;
        if (stat(path, &st) == 0) return 0;
        struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };
        nanosleep(&ts, NULL);
    }
    return -1;
}

static int send_fd(int dst, int fd) {
    char dummy = 0;
    struct iovec iov = { .iov_base = &dummy, .iov_len = 1 };
    union {
        struct cmsghdr cm;
        char buf[CMSG_SPACE(sizeof(int))];
    } u;
    memset(&u, 0, sizeof(u));
    struct msghdr mh = {0};
    mh.msg_iov = &iov;
    mh.msg_iovlen = 1;
    mh.msg_control = u.buf;
    mh.msg_controllen = sizeof(u.buf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&mh);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(sizeof(int));
    memcpy(CMSG_DATA(cmsg), &fd, sizeof(int));
    for (;;) {
        ssize_t r = sendmsg(dst, &mh, MSG_NOSIGNAL);
        if (r > 0) return 0;
        if (r < 0 && errno == EINTR) continue;
        return -1;
    }
}

static int parse_backends(const char *env, char *paths[MAX_BACKENDS]) {
    int n = 0;
    char *tmp = strdup(env);
    char *save = NULL;
    char *tok = strtok_r(tmp, ",", &save);
    while (tok && n < MAX_BACKENDS) {
        paths[n++] = strdup(tok);
        tok = strtok_r(NULL, ",", &save);
    }
    free(tmp);
    return n;
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    signal(SIGPIPE, SIG_IGN);

    int port = 9999;
    if (getenv("LB_PORT")) port = atoi(getenv("LB_PORT"));
    int backlog = 4096;
    if (getenv("LB_BACKLOG")) backlog = atoi(getenv("LB_BACKLOG"));

    const char *socks_env = getenv("API_SOCKETS");
    if (!socks_env || !*socks_env) socks_env = "/sockets/api1.sock,/sockets/api2.sock";

    char *paths[MAX_BACKENDS] = {0};
    int nb = parse_backends(socks_env, paths);
    if (nb <= 0) {
        fprintf(stderr, "[lb] sem backends\n");
        return 2;
    }

    // Espera todas as APIs estarem prontas e conecta.
    int backends[MAX_BACKENDS];
    for (int i = 0; i < nb; i++) {
        fprintf(stderr, "[lb] aguardando %s\n", paths[i]);
        if (wait_for_socket(paths[i]) < 0) {
            fprintf(stderr, "[lb] timeout aguardando %s\n", paths[i]);
            return 3;
        }
        int fd = -1;
        for (int t = 0; t < 100; t++) {
            fd = connect_backend(paths[i]);
            if (fd >= 0) break;
            struct timespec ts = { .tv_sec = 0, .tv_nsec = 100 * 1000 * 1000 };
            nanosleep(&ts, NULL);
        }
        if (fd < 0) {
            fprintf(stderr, "[lb] falha conectando %s\n", paths[i]);
            return 4;
        }
        backends[i] = fd;
        fprintf(stderr, "[lb] conectado em %s (fd=%d)\n", paths[i], fd);
    }

    // Socket de escuta TCP.
    int lfd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (lfd < 0) {
        perror("socket");
        return 5;
    }
    int on = 1;
    setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    setsockopt(lfd, SOL_SOCKET, SO_REUSEPORT, &on, sizeof(on));
    setsockopt(lfd, IPPROTO_TCP, TCP_DEFER_ACCEPT, &on, sizeof(on));

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(port);
    if (bind(lfd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind");
        return 6;
    }
    if (listen(lfd, backlog) < 0) {
        perror("listen");
        return 7;
    }

    fprintf(stderr, "[lb] escutando :%d backlog=%d, %d backends\n", port, backlog, nb);

    int rr = 0;
    for (;;) {
        int cfd = accept4(lfd, NULL, NULL, SOCK_CLOEXEC);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            continue;
        }
        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        int target = backends[rr];
        rr = (rr + 1) % nb;
        if (send_fd(target, cfd) < 0) {
            close(cfd);
            continue;
        }
        close(cfd);
    }
}
