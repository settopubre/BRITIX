#include <stdio.h>
#include <sys/mman.h>

int main() {
    printf("MAP_PRIVATE = %zu (0x%zx)\n", (size_t)MAP_PRIVATE, (size_t)MAP_PRIVATE);
    printf("MAP_SHARED = %zu (0x%zx)\n", (size_t)MAP_SHARED, (size_t)MAP_SHARED);
    printf("PROT_READ = %zu (0x%zx)\n", (size_t)PROT_READ, (size_t)PROT_READ);
    printf("PROT_WRITE = %zu (0x%zx)\n", (size_t)PROT_WRITE, (size_t)PROT_WRITE);
    return 0;
}
