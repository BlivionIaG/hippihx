#include <cstdio>

#include "hippihx/smoke.hpp"

int main() {
  const int magic = hippihx_smoke_magic();
  if (magic != HIPPIHX_SMOKE_MAGIC) {
    std::fprintf(stderr, "hippihx smoke: unexpected magic %d\n", magic);
    return 1;
  }
  std::printf("hippihx smoke ok magic=0x%x\n", magic);
  return 0;
}
