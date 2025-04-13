import React from 'react';
import styled from 'styled-components';
import { motion } from 'framer-motion';
import { Container, Heading, Text, Section } from '../styles/componentStyles';

const AboutUsContainer = styled(Container)`
  max-width: 1000px;
  text-align: center;
`;

const TeamSection = styled(motion.div)`
  margin-top: ${({ theme }) => theme.space.lg};
  padding: ${({ theme }) => theme.space.xl};
  text-align: left;
  background: ${({ theme }) => theme.colors.backgroundAlt};
  border-radius: 20px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
  overflow: hidden;
  position: relative;
  
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 5px;
    background: linear-gradient(to right, ${({ theme }) => theme.colors.primary}, ${({ theme }) => theme.colors.secondary});
  }
  
  transition: transform 0.3s ease, box-shadow 0.3s ease;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.12);
  }
`;

const SectionTitle = styled(Heading)`
  font-size: ${({ theme }) => theme.fontSizes['2xl']};
  margin-bottom: ${({ theme }) => theme.space.md};
  color: ${({ theme }) => theme.colors.primary};
  text-align: center;
  position: relative;
  display: inline-block;
  
  &::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 3px;
    background: ${({ theme }) => theme.colors.secondary};
    border-radius: 3px;
  }
`;

const MemberRole = styled.span`
    font-weight: 600;
    color: ${({ theme }) => theme.colors.primary};
    transition: color 0.2s ease;
`;

const ValuesList = styled(motion.ul)`
    padding-left: 1.2rem;
`;

const ValueItem = styled(motion.li)`
    margin-bottom: ${({ theme }) => theme.space.sm};
    position: relative;
    padding-left: 0.5rem;

    &::marker {
        color: ${({ theme }) => theme.colors.primary};
    }

    &:hover ${MemberRole} {
        color: ${({ theme }) => theme.colors.secondary};
    }
`;

const AnimatedHeading = styled(motion.div)`
    margin-bottom: ${({ theme }) => theme.space.lg};
`;

const AboutUsPage = () => {
    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                delayChildren: 0.3,
                staggerChildren: 0.2
            }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 20 },
        visible: {
            opacity: 1,
            y: 0,
            transition: { type: "spring", stiffness: 300, damping: 24 }
        }
    };

    const valueItemVariants = {
        hidden: { opacity: 0, x: -20 },
        visible: {
            opacity: 1,
            x: 0,
            transition: { type: "spring", stiffness: 100, damping: 12 }
        },
        hover: {
            x: 5,
            transition: { type: "spring", stiffness: 300, damping: 12 }
        }
    };

    return (
        <Section
            as={motion.section}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.6 }}
        >
            <AboutUsContainer>
                <AnimatedHeading
                    as={motion.div}
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{
                        duration: 0.8,
                        type: "spring",
                        stiffness: 100
                    }}
                >
                    <Heading
                        size="3xl"
                        mb="lg"
                        align="center"
                        as={motion.h1}
                        whileInView={{ scale: [0.95, 1.02, 1] }}
                        transition={{ duration: 0.8, times: [0, 0.5, 1] }}
                    >
                        Meet the Team
                    </Heading>
                    <Text
                        size="lg"
                        color="textSecondary"
                        align="center"
                        mb="xl"
                        as={motion.p}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.4, duration: 0.8 }}
                    >
                        We are a passionate and result-driven team dedicated to leveraging cutting-edge AI
                        for impactful diagnostic solutions. Performance, precision, and elegant design
                        are at the core of everything we build.
                    </Text>
                </AnimatedHeading>

                <motion.div
                    variants={containerVariants}
                    initial="hidden"
                    animate="visible"
                >
                    <TeamSection
                        variants={itemVariants}
                        whileHover={{
                            boxShadow: "0 20px 40px rgba(0, 0, 0, 0.15)",
                        }}
                    >
                        <SectionTitle size="xl" mb="lg">Our Mission</SectionTitle>
                        <Text mb="md" as={motion.p} variants={itemVariants}>
                            Our mission is to empower healthcare professionals with advanced, reliable, and
                            user-friendly tools. We believe that sophisticated AI should be accessible and
                            seamlessly integrated into clinical workflows to improve patient outcomes.
                        </Text>
                        <Text as={motion.p} variants={itemVariants}>
                            This X-Ray Diagnostic Assistant is a testament to our commitment, combining a
                            powerful deep learning model with an intuitive interface.
                        </Text>
                    </TeamSection>

                    <TeamSection
                        variants={itemVariants}
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.4, duration: 0.5 }}
                    >
                        <SectionTitle size="xl" mb="lg">Our Values</SectionTitle>
                        <ValuesList
                            variants={containerVariants}
                            initial="hidden"
                            animate="visible"
                            transition={{ staggerChildren: 0.1, delayChildren: 0.2 }}
                        >
                            <ValueItem
                                variants={valueItemVariants}
                                whileHover="hover"
                            >
                                <Text mb="sm"><MemberRole>Innovation:</MemberRole> Continuously exploring and implementing the latest advancements in AI and medical imaging.</Text>
                            </ValueItem>
                            <ValueItem
                                variants={valueItemVariants}
                                whileHover="hover"
                            >
                                <Text mb="sm"><MemberRole>Performance:</MemberRole> Ensuring our models are accurate, efficient, and reliable.</Text>
                            </ValueItem>
                            <ValueItem
                                variants={valueItemVariants}
                                whileHover="hover"
                            >
                                <Text mb="sm"><MemberRole>Elegance:</MemberRole> Crafting interfaces that are not only functional but also intuitive and aesthetically pleasing.</Text>
                            </ValueItem>
                            <ValueItem
                                variants={valueItemVariants}
                                whileHover="hover"
                            >
                                <Text mb="sm"><MemberRole>Collaboration:</MemberRole> Working closely together and valuing diverse perspectives to build the best possible product.</Text>
                            </ValueItem>
                            <ValueItem
                                variants={valueItemVariants}
                                whileHover="hover"
                            >
                                <Text><MemberRole>Impact:</MemberRole> Striving to make a real difference in the field of medical diagnostics.</Text>
                            </ValueItem>
                        </ValuesList>
                    </TeamSection>
                </motion.div>
            </AboutUsContainer>
        </Section>
    );
};

export default AboutUsPage;