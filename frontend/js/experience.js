/* ============================================
   Resume Auditor — Game Logic
   Gamified Multi-Bias Experience
   ============================================ */

document.addEventListener('DOMContentLoaded', function () {
    'use strict';

    // ---- CANDIDATE DATA ----
    // 4 candidates where AI is CORRECT (player must AGREE to score)
    // 6 candidates where AI is BIASED/WRONG (player must OVERRIDE to score)
    const candidates = [
        {
            id: 1,
            name: "Alex Vance",
            experience: "8 Years",
            skills: "React, Node.js, Systems Architecture",
            email: "alex.vance@tech.net",
            age: 30,
            gender: "MALE",
            graduationYear: "2018",
            university: "State Tech Bootcamp",
            aiRecommendation: 'REJECTED',
            isAiCorrect: false,
            aiReason: "Education credentials do not align with historical high-performer baseline.",
            biasExplanation: "The Pedigree Bias — AI incorrectly penalizes non-traditional education pathways, favoring Ivy League over actual, proven job experience."
        },
        {
            id: 2,
            name: "Diana Rossi",
            experience: "12 Years",
            skills: "React, Node.js, PostgreSQL",
            email: "diana.r@aol.com",
            age: 49,
            gender: "FEMALE",
            graduationYear: "1998",
            university: "Central State University",
            aiRecommendation: 'REJECTED',
            isAiCorrect: false,
            aiReason: "Applicant details (email domain, grad year) flag as low culture-add probability.",
            biasExplanation: "The Ageism Proxy — AI learned to correlate older graduation dates and legacy email providers with 'culture mismatch,' a proxy for illegal age discrimination."
        },
        {
            id: 3,
            name: "Jake Miller",
            experience: "1 Year",
            skills: "HTML, CSS",
            email: "jmiller@college.edu",
            age: 23,
            gender: "MALE",
            graduationYear: "2025",
            university: "Top Tier U",
            aiRecommendation: 'REJECTED',
            isAiCorrect: true,
            aiReason: "Candidate lacks requisite senior-level backend frameworks.",
            biasExplanation: "The Legitimate Reject — A genuinely unqualified candidate. The AI worked correctly."
        },
        {
            id: 4,
            name: "William Hastings",
            experience: "1 Year",
            skills: "Python Basics, Equestrian, Lacrosse",
            email: "w.hastings@harvard.edu",
            age: 24,
            gender: "MALE",
            graduationYear: "2024",
            university: "Harvard",
            aiRecommendation: 'ACCEPTED',
            isAiCorrect: false,
            aiReason: "Extracurricular keywords highly correlate with successful executive traits.",
            biasExplanation: "The Affinity / Class Bias — The AI over-indexes on affluent hobbies (Equestrian / Lacrosse) and Ivy League keywords, ignoring the actual lack of technical skills (often called the 'Bro-Culture' proxy)."
        },
        {
            id: 5,
            name: "Maya Patel",
            experience: "6 Years (2-year Sabbatical)",
            skills: "Java, AWS, Kubernetes, CI/CD",
            email: "mpatel.dev@gmail.com",
            age: 32,
            gender: "FEMALE",
            graduationYear: "2015",
            university: "University of Michigan",
            aiRecommendation: 'REJECTED',
            isAiCorrect: false,
            aiReason: "Discontinuous work history detected; flagged as high attrition risk.",
            biasExplanation: "The Gender / Caregiver Penalty — The AI penalizes a 2-year 'Career Break', which disproportionately filters out women who take maternity leave or individuals acting as family caregivers."
        },
        {
            id: 6,
            name: "David Chen",
            experience: "5 Years",
            skills: "Go, Docker, Microservices, MongoDB",
            email: "david.chen@gmail.com",
            age: 28,
            gender: "MALE",
            graduationYear: "2021",
            university: "Georgia Tech",
            aiRecommendation: 'ACCEPTED',
            isAiCorrect: true,
            aiReason: "Metrics indicate high competency alignment with required technical benchmarks.",
            biasExplanation: "The Legitimate Accept — A highly qualified candidate that the AI correctly identified."
        },
        {
            id: 7,
            name: "Tariq Al-Fayed",
            experience: "6 Years Top International Firm",
            skills: "Python, Django, React, Cloud",
            email: "tariq.alfayed@dev.com",
            age: 29,
            gender: "MALE",
            graduationYear: "2020",
            university: "Cairo University",
            aiRecommendation: 'REJECTED',
            isAiCorrect: false,
            aiReason: "Applicant origins deviate from established high-conversion historical parameters.",
            biasExplanation: "The Name/Origin Bias — The AI's training data historically favored Western-sounding names and US-based universities, resulting in xenophobic filtering despite top international tech firm experience."
        },
        {
            id: 8,
            name: "Sarah Jenkins",
            experience: "7 Years (Collaborated, Supported)",
            skills: "Ruby on Rails, PostgreSQL, Redis",
            email: "s.jenkins@fastmail.com",
            age: 31,
            gender: "FEMALE",
            graduationYear: "2019",
            university: "Boston University",
            aiRecommendation: 'REJECTED',
            isAiCorrect: false,
            aiReason: "Leadership indicator metrics fall below semantic threshold matrix.",
            biasExplanation: "The Gendered Language Bias — The AI filters for aggressive, historically male-coded verbs ('Spearheaded'), rejecting qualified candidates who use collaborative language ('Supported,' 'Facilitated')."
        },
        {
            id: 9,
            name: "Chris Evans",
            experience: "0 Years",
            skills: "C, C++, Java, JS, Python, Rust, Go, SQL, AWS, Azure, GCP",
            email: "cevans.tech@gmail.com",
            age: 22,
            gender: "MALE",
            graduationYear: "2026",
            university: "State College",
            aiRecommendation: 'ACCEPTED',
            isAiCorrect: false,
            aiReason: "Exceptional multi-domain capability detected; skill density exceeds top quintile.",
            biasExplanation: "The Keyword Stuffer — The AI is easily manipulated by keyword stuffing, failing to correlate massive listed skills with actual 0-year work history."
        },
        {
            id: 10,
            name: "Elena Rodriguez",
            experience: "10 Years",
            skills: "TypeScript, React, Node.js, GraphQL, AWS",
            email: "erodriguez@dev.io",
            age: 34,
            gender: "FEMALE",
            graduationYear: "2016",
            university: "Stanford University",
            aiRecommendation: 'ACCEPTED',
            isAiCorrect: true,
            aiReason: "Optimal alignment across all vector heuristics and technical prerequisites.",
            biasExplanation: "The Perfect Match — The AI correctly passed a strong candidate with 10 years experience and a flawless tech stack."
        }
    ];

    // ---- STATE VARIABLES ----
    let currentCandidateIndex = 0;
    let timeLeft = 60;
    let timerInterval = null;
    let animationTimeout = null;
    let userResults = []; 
    let isAnimating = false;
    let correctCount = 0;

    // ---- DOM REFS ----
    const stateEls = {
        briefing: document.getElementById('state-briefing'),
        instructions: document.getElementById('state-instructions'),
        game: document.getElementById('state-game'),
        diagnostic: document.getElementById('state-diagnostic'),
        reward: document.getElementById('state-reward'),
        playbook: document.getElementById('state-playbook')
    };

    const gameOverScreen = document.getElementById('game-over-screen');
    const timerEl = document.getElementById('timer-display');
    const progressHeaderEl = document.getElementById('candidate-progress-header');
    const progressEl = document.getElementById('candidate-progress');

    // Resume Module DOM refs
    const domEls = {
        badge: document.getElementById('ai-badge'),
        name: document.getElementById('candidate-name'),
        experience: document.getElementById('candidate-experience'),
        skills: document.getElementById('candidate-skills'),
        email: document.getElementById('candidate-email'),
        age: document.getElementById('candidate-age'),
        gender: document.getElementById('candidate-gender'),
        grad: document.getElementById('candidate-grad'),
        university: document.getElementById('candidate-university')
    };

    const btnAgree = document.getElementById('btn-agree');
    const btnOverride = document.getElementById('btn-override');
    const btnDeployPatch = document.getElementById('btn-deploy-patch');
    const btnUnlockPlaybook = document.getElementById('btn-unlock-playbook');

    const scoreReviewed = document.getElementById('score-reviewed');
    const scoreAgreed = document.getElementById('score-agreed');
    const scoreOverridden = document.getElementById('score-overridden');

    const audioAccepted = document.getElementById('audio-accepted');
    const audioRejected = document.getElementById('audio-rejected');
    const audioTimer = document.getElementById('audio-timer');

    // Action Handlers
    const handleAgree = () => handleDecision(true);
    const handleOverride = () => handleDecision(false);

    function switchState(stateName) {
        Object.values(stateEls).forEach(el => {
            if(el) el.classList.remove('active');
        });
        if (gameOverScreen) gameOverScreen.style.display = 'none';
        if (stateEls[stateName]) stateEls[stateName].classList.add('active');
    }

    // Fisher-Yates shuffle
    function shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
    }

    // INIT FUNCTION
    function initGame() {
        // Clear lingering timers
        if (timerInterval) clearInterval(timerInterval);
        if (animationTimeout) clearTimeout(animationTimeout);

        // Reset state variables
        currentCandidateIndex = 0;
        timeLeft = 60;
        userResults = [];
        correctCount = 0;
        isAnimating = false;

        // Initialize Controls
        initControls();

        // UI Prep
        switchState('game');
        if (audioTimer) {
            audioTimer.currentTime = 0;
            audioTimer.play().catch(e => console.warn('Timer playback failed:', e));
        }

        // Setup deck
        shuffleArray(candidates);
        renderCandidate();

        // Start timer
        if (timerEl) timerEl.textContent = `TIME: ${timeLeft}s`;
        timerInterval = setInterval(() => {
            timeLeft--;
            if (timerEl) timerEl.textContent = `TIME: ${timeLeft}s`;
            if (timeLeft <= 0) {
                endGame();
            }
        }, 1000);
    }
    
    function initControls() {
        // Robust listener initialization
        if (btnAgree) {
            btnAgree.removeEventListener('click', handleAgree);
            btnAgree.addEventListener('click', handleAgree);
        }
        if (btnOverride) {
            btnOverride.removeEventListener('click', handleOverride);
            btnOverride.addEventListener('click', handleOverride);
        }
    }

    const biasAudio = document.getElementById('bias-audio');
    const instructionsAudio = document.getElementById('instructions-audio');

    if (biasAudio) {
        biasAudio.addEventListener('ended', () => {
            switchState('instructions');
            if (instructionsAudio) {
                instructionsAudio.currentTime = 0;
                instructionsAudio.play().catch(e => console.warn('Instructions audio playback failed:', e));
            }
        });
    }

    if (instructionsAudio) {
        instructionsAudio.addEventListener('ended', initGame);
    }

    function renderCandidate() {
        if (currentCandidateIndex >= candidates.length) return;

        const candidate = candidates[currentCandidateIndex];

        if (progressHeaderEl) progressHeaderEl.textContent = `${currentCandidateIndex + 1}/${candidates.length}`;
        if (progressEl) progressEl.textContent = `PROGRESS: ${currentCandidateIndex + 1}/${candidates.length}`;

        if (domEls.name) domEls.name.textContent = candidate.name;
        if (domEls.experience) domEls.experience.innerHTML = candidate.experience;
        if (domEls.skills) domEls.skills.textContent = candidate.skills;
        if (domEls.email) domEls.email.textContent = candidate.email;
        if (domEls.age) domEls.age.textContent = candidate.age;
        if (domEls.gender) domEls.gender.textContent = candidate.gender;
        if (domEls.grad) domEls.grad.textContent = candidate.graduationYear;
        if (domEls.university) domEls.university.textContent = candidate.university;

        if (domEls.badge) {
            domEls.badge.textContent = candidate.aiRecommendation;
            domEls.badge.className = 'ai-decision-banner';
            if (candidate.aiRecommendation === 'ACCEPTED') {
                domEls.badge.classList.add('badge-accepted');
            } else {
                domEls.badge.classList.add('badge-rejected');
            }
        }
        
        const aiReasonEl = document.getElementById('ai-reason-text');
        if (aiReasonEl) aiReasonEl.textContent = candidate.aiReason;

        if (audioAccepted && audioRejected) {
            audioAccepted.pause();
            audioAccepted.currentTime = 0;
            audioRejected.pause();
            audioRejected.currentTime = 0;

            if (candidate.aiRecommendation === 'ACCEPTED') {
                audioAccepted.play().catch(err => console.warn('Playback failed:', err));
            } else {
                audioRejected.play().catch(err => console.warn('Playback failed:', err));
            }
        }

        const card = document.getElementById('card-container');
        if (card) {
            // Explicitly force CSS reflow for smooth consecutive animations
            card.classList.remove('swipe-left', 'swipe-right', 'fade-in', 'slide-up');
            void card.offsetWidth; // Trigger reflow
            card.classList.add('fade-in'); 
            card.style.opacity = '1';
        }
    }

    function handleDecision(userAgreed) {
        // Animation Guard
        if (isAnimating) return;
        isAnimating = true;

        const card = document.getElementById('card-container');
        if (!card) { isAnimating = false; return; }

        const candidate = candidates[currentCandidateIndex];

        const isCorrect = (userAgreed === candidate.isAiCorrect);
        if (isCorrect) correctCount++;

        userResults.push({
            candidateId: candidate.id,
            agreed: userAgreed,
            wasAiCorrect: candidate.isAiCorrect,
            correct: isCorrect
        });

        card.classList.add(userAgreed ? 'swipe-right' : 'swipe-left');

        animationTimeout = setTimeout(() => {
            currentCandidateIndex++;
            if (currentCandidateIndex < candidates.length) {
                renderCandidate();
                isAnimating = false;
            } else {
                endGame();
            }
        }, 400);
    }

    function endGame() {
        if (timerInterval) clearInterval(timerInterval);
        
        if (audioTimer) {
            audioTimer.pause();
            audioTimer.currentTime = 0;
        }

        const reviewed = userResults.length;
        const agreed = userResults.filter(d => d.agreed).length;
        const overridden = userResults.filter(d => !d.agreed).length;

        if (correctCount >= 8) {
            if (scoreReviewed) scoreReviewed.textContent = `${reviewed}/${candidates.length}`;
            if (scoreAgreed) scoreAgreed.textContent = agreed;
            if (scoreOverridden) scoreOverridden.textContent = overridden;
            switchState('diagnostic');
        } else {
            showGameOver();
        }
    }

    function showGameOver() {
        Object.values(stateEls).forEach(el => {
            if(el) el.classList.remove('active');
        });

        if (gameOverScreen) {
            gameOverScreen.style.display = 'flex';
        }

        function reloadHandler() {
            document.removeEventListener('click', reloadHandler);
            initGame(); // explicitly calls initialization logic on retry
        }
        setTimeout(() => {
            document.addEventListener('click', reloadHandler);
        }, 300);
    }

    document.querySelectorAll('#state-diagnostic .feature-item input[type="checkbox"]').forEach(cb => {
        cb.addEventListener('change', (e) => {
            const parentItem = e.target.closest('.feature-item');
            if (e.target.checked) {
                parentItem.classList.remove('unchecked');
            } else {
                parentItem.classList.add('unchecked');
            }
        });
    });

    if (btnDeployPatch) {
        btnDeployPatch.addEventListener('click', () => {
            const isExpValid = document.getElementById('feature-exp')?.checked === true;
            const isSkillsValid = document.getElementById('feature-skills')?.checked === true;
            
            const isNameValid = document.getElementById('feature-name')?.checked === false;
            const isGenderValid = document.getElementById('feature-gender')?.checked === false;
            const isEmailValid = document.getElementById('feature-email')?.checked === false;
            const isGradValid = document.getElementById('feature-grad')?.checked === false;
            const isUnivValid = document.getElementById('feature-univ')?.checked === false;
            
            if (isExpValid && isSkillsValid && isNameValid && isGenderValid && isEmailValid && isGradValid && isUnivValid) {
                spawnConfetti();
                switchState('reward');
            } else {
                showGameOver();
            }
        });
    }

    if (btnUnlockPlaybook) {
        btnUnlockPlaybook.addEventListener('click', (e) => {
            e.preventDefault();
            switchState('playbook');
        });
    }

    function spawnConfetti() {
        const container = document.createElement('div');
        container.className = 'confetti-container';
        document.body.appendChild(container);
        const colors = ['#7c3aed', '#22c55e', '#f59e0b', '#ef4444', '#3b82f6', '#ec4899'];
        const shapes = ['square', 'circle'];
        for (let i = 0; i < 80; i++) {
            const piece = document.createElement('div');
            piece.className = 'confetti-piece';
            const color = colors[Math.floor(Math.random() * colors.length)];
            const shape = shapes[Math.floor(Math.random() * shapes.length)];
            const left = Math.random() * 100;
            const size = Math.random() * 8 + 6;
            const duration = Math.random() * 2 + 2;
            const delay = Math.random() * 1;
            piece.style.cssText = `
                left: ${left}%;
                width: ${size}px;
                height: ${size}px;
                background: ${color};
                border-radius: ${shape === 'circle' ? '50%' : '2px'};
                animation-duration: ${duration}s;
                animation-delay: ${delay}s;
            `;
            container.appendChild(piece);
        }
        setTimeout(() => container.remove(), 4500);
    }

    document.addEventListener('keydown', (e) => {
        // Prevent accidental rapid-fire inputs if a key is held down
        if (e.repeat) return;

        if (e.shiftKey && (e.key === 'S' || e.key === 's')) {
            if (stateEls.briefing && stateEls.briefing.classList.contains('active')) {
                console.log("DEV SHORTCUT: Briefing skipped.");
                if (biasAudio) biasAudio.pause();
                switchState('instructions');
                if (instructionsAudio) {
                    instructionsAudio.currentTime = 0;
                    instructionsAudio.play().catch(e => console.warn('Instructions audio playback failed:', e));
                }
                return;
            }
            if (stateEls.instructions && stateEls.instructions.classList.contains('active')) {
                console.log("DEV SHORTCUT: Instructions skipped.");
                if (instructionsAudio) { instructionsAudio.pause(); instructionsAudio.currentTime = 0; }
                initGame();
                return;
            }
        }

        if (!stateEls.game || !stateEls.game.classList.contains('active')) return;
        
        // Secondary Guard: ignore keys if an animation is currently playing
        if (isAnimating) return;

        if (e.key === 'ArrowLeft') {
            handleDecision(false); 
        } else if (e.key === 'ArrowRight') {
            handleDecision(true);  
        }
    });
});
