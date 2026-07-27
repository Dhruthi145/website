const header = document.querySelector('.site-header');
const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');
const dropdownToggle = document.querySelector('.dropdown-toggle');
const dropdownMenu = document.querySelector('.dropdown-menu');
const revealItems = document.querySelectorAll('.reveal, .reveal-left, .reveal-right');
const counters = document.querySelectorAll('[data-count]');
const businessCards = document.querySelectorAll('.business-card[data-href]');

const setHeaderState = () => {
  if (window.scrollY > 24) {
    header.classList.add('scrolled');
  } else {
    header.classList.remove('scrolled');
  }
};

const toggleMenu = () => {
  const open = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
};

const closeDropdown = () => {
  dropdownToggle?.classList.remove('active');
  dropdownToggle?.setAttribute('aria-expanded', 'false');
  dropdownMenu?.classList.remove('open');
};

const toggleDropdown = () => {
  const expanded = dropdownToggle?.getAttribute('aria-expanded') === 'true';
  if (expanded) {
    closeDropdown();
  } else {
    dropdownToggle?.classList.add('active');
    dropdownToggle?.setAttribute('aria-expanded', 'true');
    dropdownMenu?.classList.add('open');
  }
};

const revealOnScroll = () => {
  revealItems.forEach((item) => {
    const rect = item.getBoundingClientRect();
    if (rect.top < window.innerHeight - 90) {
      item.classList.add('visible');
    }
  });
};

const animateCounters = () => {
  counters.forEach((counter) => {
    const target = Number(counter.getAttribute('data-count'));
    const duration = 1400;
    const startTime = performance.now();

    const step = (now) => {
      const progress = Math.min((now - startTime) / duration, 1);
      const value = Math.floor(progress * target);
      counter.textContent = `${value}`;
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        counter.textContent = `${target}`;
      }
    };

    requestAnimationFrame(step);
  });
};

window.addEventListener('scroll', () => {
  setHeaderState();
  revealOnScroll();
});

navToggle?.addEventListener('click', toggleMenu);
dropdownToggle?.addEventListener('click', (event) => {
  event.stopPropagation();
  toggleDropdown();
});

navLinks?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
    closeDropdown();
  });
});

businessCards.forEach((card) => {
  card.addEventListener('click', () => {
    window.location.href = card.getAttribute('data-href');
  });
});

document.addEventListener('click', (event) => {
  if (!event.target.closest('.nav-dropdown')) {
    closeDropdown();
  }
});

window.addEventListener('load', () => {
  setHeaderState();
  revealOnScroll();
  animateCounters();
});

const heroVisual = document.querySelector('.hero-visual');
if (heroVisual) {
  heroVisual.addEventListener('mousemove', (event) => {
    const rect = heroVisual.getBoundingClientRect();
    const x = (event.clientX - rect.left) / rect.width - 0.5;
    const y = (event.clientY - rect.top) / rect.height - 0.5;
    heroVisual.style.transform = `perspective(1200px) rotateY(${x * 6}deg) rotateX(${y * -6}deg)`;
  });

  heroVisual.addEventListener('mouseleave', () => {
    heroVisual.style.transform = 'perspective(1200px) rotateY(0deg) rotateX(0deg)';
  });
}
