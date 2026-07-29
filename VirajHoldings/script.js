document.addEventListener('DOMContentLoaded', () => {

  /* ---------- Splash screen: show for ~2.5s, then reveal site ---------- */
  const splash = document.getElementById('splash');

  window.setTimeout(() => {
    splash.classList.add('is-hidden');
    document.body.classList.add('ready');
  }, 2500);

  window.setTimeout(() => {
    splash.remove();
  }, 3400);

  /* ---------- Footer year ---------- */
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  /* ---------- Mobile nav toggle ---------- */
  const navToggle = document.getElementById('nav-toggle');
  const siteNav = document.getElementById('site-nav');

  navToggle.addEventListener('click', () => {
    const isOpen = siteNav.classList.toggle('is-open');
    navToggle.setAttribute('aria-expanded', String(isOpen));
  });

  siteNav.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', () => {
      siteNav.classList.remove('is-open');
      navToggle.setAttribute('aria-expanded', 'false');
      showHome();
    });
  });

  /* ---------- View switching: home <-> business detail pages ---------- */
  const homeView = document.getElementById('home-view');
  const productDetails = document.querySelectorAll('.product-detail');
  const productCards = document.querySelectorAll('.product-card');

  function showHome(){
    homeView.style.display = '';
    productDetails.forEach(d => d.classList.remove('is-active'));
  }

  function showProduct(key){
    homeView.style.display = 'none';
    productDetails.forEach(d => {
      d.classList.toggle('is-active', d.dataset.detail === key);
    });
    window.scrollTo({ top: 0, behavior: 'instant' in window ? 'instant' : 'auto' });
  }

  productCards.forEach(card => {
    card.addEventListener('click', () => {
      showProduct(card.dataset.product);
    });
  });

  /* ---------- Footer "Business Verticals" links open the matching detail page ---------- */
  document.querySelectorAll('[data-product-link]').forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      showProduct(link.dataset.productLink);
    });
  });

  document.querySelectorAll('.back-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      showHome();
      const businessesSection = document.getElementById('businesses');
      businessesSection.scrollIntoView({ behavior: 'smooth' });
    });
  });

  /* ---------- Brand logo click returns home ---------- */
  document.querySelector('.brand').addEventListener('click', (e) => {
    e.preventDefault();
    showHome();
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

});